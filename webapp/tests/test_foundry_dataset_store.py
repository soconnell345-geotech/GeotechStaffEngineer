"""Offline tests for the Foundry dataset store (BACKLOG INF-38).

A fake ``foundry_sdk`` client stands in for the platform SDK: no network,
and every RID, host, token and name below is made up. The fake keeps the
rules the AI FDE described (2026-11-03): files are written inside a
transaction, a branch has one open transaction at a time, a listing returns
files only, and a path is written once per transaction.
"""

import json
import os
import sys
import time
import types
from datetime import datetime, timezone

import pytest

import webapp.core as core
import webapp.foundry_dataset_store as fds
import webapp.sharepoint_store as sp

RID = "ri.foundry.main.dataset.00000000-made-up-0000-000000000000"
OTHER_RID = "ri.foundry.main.dataset.11111111-made-up-1111-111111111111"

_ENVS = (fds.ENV_DATASET, fds.ENV_BRANCH, fds.SERVICE_DISCOVERY_ENV,
         "GEOTECH_DEPLOYMENT", "GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN",
         "GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME", "FOUNDRY_URL",
         "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "MAESTRO_CERT_PEM_PATH",
         "FOUNDRY_CLI_CERT_PEM_PATH",
         sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET, sp.ENV_TOKEN,
         sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for e in _ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(sp, "_graph_configured", lambda: False)
    yield


# ---------------------------------------------------------------------------
# the fake SDK client
# ---------------------------------------------------------------------------

class OpenTransactionAlreadyExists(Exception):
    """Named like the SDK's error, which is how the store recognises it."""


class FileAlreadyExists(Exception):
    pass


class FakeDataset:
    """One dataset's committed files and its open transaction."""

    def __init__(self, conflicts=0):
        self.files = {}              # path -> bytes (committed)
        self.open = None             # rid of the open transaction
        self.staged = {}             # rid -> {path: bytes}
        self.conflicts = conflicts   # creates refused before one succeeds
        self.calls = []              # ("create"|"upload"|..., detail)
        self.n = 0
        self.fail_commit = False

    # Dataset.Transaction -------------------------------------------------
    def create(self, dataset_rid, *, transaction_type, branch_name=None):
        self.calls.append(("create", transaction_type, branch_name))
        if self.conflicts or self.open is not None:
            if self.conflicts:
                self.conflicts -= 1
            raise OpenTransactionAlreadyExists("a transaction is open")
        self.n += 1
        self.open = f"ri.foundry.main.transaction.fake-{self.n}"
        self.staged[self.open] = {}
        return types.SimpleNamespace(rid=self.open, status="OPEN")

    def commit(self, dataset_rid, transaction_rid):
        self.calls.append(("commit", transaction_rid))
        assert transaction_rid == self.open
        if self.fail_commit:
            raise RuntimeError("commit refused")
        self.files.update(self.staged.pop(transaction_rid))
        self.open = None

    def abort(self, dataset_rid, transaction_rid):
        self.calls.append(("abort", transaction_rid))
        if transaction_rid == self.open:
            self.staged.pop(transaction_rid, None)
            self.open = None

    # Dataset.File -------------------------------------------------------
    def upload(self, dataset_rid, file_path, body, *, transaction_rid=None):
        self.calls.append(("upload", file_path, transaction_rid))
        assert not file_path.startswith("/")
        if transaction_rid != self.open:
            raise RuntimeError("TransactionNotOpen")
        if file_path in self.staged[transaction_rid]:
            raise FileAlreadyExists(file_path)
        self.staged[transaction_rid][file_path] = bytes(body)

    def list(self, dataset_rid, *, branch_name=None, page_size=None,
             path_prefix=None):
        self.calls.append(("list", path_prefix))
        stamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
        return iter([types.SimpleNamespace(
            path=p, size_bytes=len(b), updated_time=stamp,
            transaction_rid="ri.foundry.main.transaction.fake")
            for p, b in sorted(self.files.items())
            if not path_prefix or p.startswith(path_prefix)])

    def content(self, dataset_rid, file_path, *, branch_name=None):
        self.calls.append(("content", file_path))
        if file_path not in self.files:
            raise RuntimeError(f"FileNotFound: {file_path}")
        return self.files[file_path]

    def count(self, kind):
        return sum(1 for c in self.calls if c[0] == kind)


class _Files:
    def __init__(self, ds):
        self.list, self.upload, self.content = ds.list, ds.upload, ds.content


class _Transactions:
    def __init__(self, ds):
        self.create, self.commit, self.abort = ds.create, ds.commit, ds.abort


def fake_client(ds):
    dataset = types.SimpleNamespace(File=_Files(ds),
                                    Transaction=_Transactions(ds))
    return types.SimpleNamespace(datasets=types.SimpleNamespace(
        Dataset=dataset))


def _fm(ds, **kw):
    kw.setdefault("sleep", lambda s: None)
    return fds.DatasetFileManager(RID, client=fake_client(ds), **kw)


def _conversation(root, tid, title, created):
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
    return conv


_CREATED = time.mktime((2026, 1, 2, 12, 0, 0, 0, 0, -1))


# ---------------------------------------------------------------------------
# the file manager alone
# ---------------------------------------------------------------------------

def test_ls_makes_folders_from_file_paths():
    ds = FakeDataset()
    ds.files = {"GSE/conversations/Alpha_2026-01-02/meta.json": b"{}",
                "GSE/conversations/Alpha_2026-01-02/files/a.pdf": b"%PDF",
                "GSE/conversations/Beta_2026-01-01/meta.json": b"{}",
                "GSE/conversations/stray.txt": b"x",
                "GSE/other/z.txt": b"z"}
    fm = _fm(ds)
    top = {(e["name"], e["type"]) for e in fm.ls("GSE/conversations")}
    assert top == {("Alpha_2026-01-02", "folder"),
                   ("Beta_2026-01-01", "folder"), ("stray.txt", "file")}
    alpha = {(e["name"], e["type"], e["path"])
             for e in fm.ls("/GSE/conversations/Alpha_2026-01-02/")}
    assert alpha == {
        ("meta.json", "file", "GSE/conversations/Alpha_2026-01-02/meta.json"),
        ("files", "folder", "GSE/conversations/Alpha_2026-01-02/files")}
    assert fm.ls("GSE/conversations/Nope") == []
    # each listing asked for the folder's own prefix, never the whole dataset
    assert [c[1] for c in ds.calls if c[0] == "list"] == [
        "GSE/conversations", "GSE/conversations/Alpha_2026-01-02",
        "GSE/conversations/Nope"]
    deep = [e["path"] for e in fm.ls("GSE/conversations", recursive=True,
                                     type="files")]
    assert len(deep) == 4 and "GSE/other/z.txt" not in deep
    assert fm.create_folder("GSE/x/y")["path"] == "GSE/x/y"   # a no-op
    assert ds.count("create") == 0
    assert fm.get_web_url("GSE/conversations") == ""


def test_an_sdk_without_path_prefix_lists_all_and_filters(tmp_path):
    ds = FakeDataset()
    ds.files = {"GSE/a/1.txt": b"1", "GSE/b/2.txt": b"2"}
    plain = ds.list

    def list_without_prefix(dataset_rid, *, branch_name=None, page_size=None,
                            **kw):
        if kw:
            raise TypeError("unexpected keyword argument 'path_prefix'")
        return plain(dataset_rid, branch_name=branch_name,
                     page_size=page_size)
    client = fake_client(ds)
    client.datasets.Dataset.File.list = list_without_prefix
    fm = fds.DatasetFileManager(RID, client=client)
    assert [e["name"] for e in fm.ls("GSE/a")] == ["1.txt"]
    assert [e["name"] for e in fm.ls("GSE/b")] == ["2.txt"]


def test_single_upload_overwrite_and_download(tmp_path):
    ds = FakeDataset()
    fm = _fm(ds)
    local = tmp_path / "note.txt"
    local.write_bytes(b"first")
    assert fm.upload_file(str(local), "GSE/n/note.txt", overwrite=True)
    local.write_bytes(b"second")
    assert fm.upload_file(str(local), "GSE/n/note.txt", overwrite=True)
    assert ds.files == {"GSE/n/note.txt": b"second"}          # newest wins
    assert ds.count("create") == ds.count("commit") == 2
    assert fm.download_file("GSE/n/note.txt") == b"second"
    out = tmp_path / "back" / "note.txt"
    assert fm.download_file("GSE/n/note.txt", local_path=str(out),
                            return_bytes=False, overwrite=True) is True
    assert out.read_bytes() == b"second"


def test_a_batch_is_one_transaction_and_opens_none_when_empty(tmp_path):
    ds = FakeDataset()
    fm = _fm(ds)
    fm.begin_batch()
    assert fm.end_batch() == 0
    assert ds.calls == []                       # nothing changed, no save
    paths = []
    for i in range(3):
        f = tmp_path / f"f{i}.txt"
        f.write_bytes(b"x" * i)
        paths.append(f"GSE/c/f{i}.txt")
        if i == 0:
            fm.begin_batch()
        fm.upload_file(str(f), paths[-1], overwrite=True)
    assert ds.files == {}                       # nothing visible yet
    assert fm.end_batch() == 3
    assert sorted(ds.files) == paths
    assert ds.count("create") == ds.count("commit") == 1
    assert len({c[2] for c in ds.calls if c[0] == "upload"}) == 1


def test_a_listing_is_fetched_once_per_batch():
    ds = FakeDataset()
    ds.files = {"GSE/c/T/meta.json": b"{}", "GSE/c/T/files/a.pdf": b"%",
                "GSE/c/T/files/sub/b.png": b"p"}
    fm = _fm(ds)
    fm.begin_batch()
    assert sp._walk_remote(fm, "GSE/c/T") and ds.count("list") == 1
    fm.end_batch()
    sp._walk_remote(fm, "GSE/c/T")                 # outside a batch: fresh
    assert ds.count("list") > 2


def test_a_batch_belongs_to_its_thread(tmp_path):
    """One process serves every viewer: a restore beginning and ending its
    batch in another session's thread leaves this session's save open."""
    import threading
    ds = FakeDataset()
    fm = _fm(ds)
    f = tmp_path / "a.txt"
    f.write_bytes(b"a")
    fm.begin_batch()
    fm.upload_file(str(f), "GSE/a.txt", overwrite=True)

    def other_session():
        fm.begin_batch()
        fm.ls("GSE")
        fm.end_batch(commit=False)
    t = threading.Thread(target=other_session)
    t.start()
    t.join()
    g = tmp_path / "b.txt"
    g.write_bytes(b"b")
    fm.upload_file(str(g), "GSE/b.txt", overwrite=True)
    assert fm.end_batch() == 2
    assert ds.files == {"GSE/a.txt": b"a", "GSE/b.txt": b"b"}
    assert ds.count("create") == 1 and ds.count("abort") == 0


def test_a_conflict_is_retried_with_a_growing_pause(tmp_path):
    ds = FakeDataset(conflicts=2)
    pauses = []
    fm = _fm(ds, sleep=pauses.append)
    f = tmp_path / "a.txt"
    f.write_bytes(b"a")
    assert fm.upload_file(str(f), "GSE/a.txt", overwrite=True)
    assert pauses == [0.5, 1.0] and ds.count("create") == 3
    assert ds.files == {"GSE/a.txt": b"a"}


def test_the_sdks_generic_conflict_counts_by_its_name(tmp_path):
    class ConflictError(Exception):
        name = "OpenTransactionAlreadyExists"
    assert fds._is_open_transaction_conflict(ConflictError("409"))
    assert not fds._is_open_transaction_conflict(RuntimeError("403"))


def test_a_conflict_that_lasts_gives_up_after_the_last_try(tmp_path):
    ds = FakeDataset(conflicts=99)
    pauses = []
    fm = _fm(ds, sleep=pauses.append)
    f = tmp_path / "a.txt"
    f.write_bytes(b"a")
    with pytest.raises(OpenTransactionAlreadyExists):
        fm.upload_file(str(f), "GSE/a.txt", overwrite=True)
    assert len(pauses) == fds.OPEN_TRIES - 1
    assert ds.count("create") == fds.OPEN_TRIES


# ---------------------------------------------------------------------------
# through the store: mirror, list, restore
# ---------------------------------------------------------------------------

def _store(ds, **kw):
    return sp.SharePointStore(file_manager=_fm(ds, **kw))


def test_the_store_names_foundry_storage_and_its_own_root():
    store = _store(FakeDataset())
    assert store.backend == "foundry" and store.label == "Foundry storage"
    assert store.root() == fds.DEFAULT_ROOT
    assert store.conversations_base() == "GeotechStaffEngineer/conversations"
    assert store.conversations_base("Jane Doe") == \
        "GeotechStaffEngineer/conversations/Jane_Doe"


def test_a_mirror_sends_its_files_in_one_transaction(tmp_path):
    ds = FakeDataset()
    store = _store(ds)
    root = str(tmp_path / "container")
    conv = _conversation(root, "T-ONE", "Footing check", _CREATED)
    up = store.mirror_conversation("T-ONE", root=root)
    assert not up["errors"], up
    assert up["uploaded"] == 4 and up["web_url"] == ""
    assert up["folder"] == ("GeotechStaffEngineer/conversations/"
                            "Footing_check_2026-01-02")
    assert ds.count("create") == ds.count("commit") == 1
    assert sorted(p.rsplit("/", 1)[-1] for p in ds.files) == [
        "calc.pdf", "figure.png", "meta.json", "transcript.jsonl"]
    assert not any(sp.MANIFEST_NAME in p for p in ds.files)

    # nothing changed: no transaction at all
    again = store.mirror_conversation("T-ONE", root=root)
    assert again["uploaded"] == 0 and again["skipped"] == 4
    assert ds.count("create") == 1

    # one file changed: it alone, overwritten, in one more transaction
    pdf = os.path.join(conv, "files", "calc.pdf")
    with open(pdf, "wb") as fh:
        fh.write(b"%PDF revised")
    os.utime(pdf, (time.time() + 5, time.time() + 5))
    third = store.mirror_conversation("T-ONE", root=root)
    assert third["uploaded"] == 1 and not third["errors"], third
    assert ds.count("create") == 2
    assert ds.files[f"{up['folder']}/files/calc.pdf"] == b"%PDF revised"


def test_the_restore_round_trip(tmp_path):
    ds = FakeDataset()
    store = _store(ds)
    before = str(tmp_path / "before_rebuild")
    after = str(tmp_path / "after_rebuild")
    conv_a = _conversation(before, "T-RT", "Wall design", _CREATED)
    assert not store.mirror_conversation("T-RT", root=before)["errors"]

    listing = store.list_remote_conversations()
    assert [r["name"] for r in listing] == ["Wall_design_2026-01-02"]
    lists_before = ds.count("list")
    res = store.restore_conversation("Wall_design_2026-01-02", root=after)
    assert res["status"] == "restored" and res["thread_id"] == "T-RT", res
    assert res["downloaded"] == 4 and not res["errors"]
    assert ds.count("list") - lists_before == 1        # listed once
    conv_b = core.conversation_dir("T-RT", after)
    for rel in ("meta.json", "transcript.jsonl", "files/calc.pdf",
                "files/figure.png"):
        with open(os.path.join(conv_a, *rel.split("/")), "rb") as a, \
                open(os.path.join(conv_b, *rel.split("/")), "rb") as b:
            assert a.read() == b.read(), rel
    assert [m["thread_id"] for m in core.list_conversations(after)] == ["T-RT"]
    # the restored manifest is complete: the next sync sends nothing
    again = store.mirror_conversation("T-RT", root=after)
    assert again["uploaded"] == 0 and not again["errors"], again


def test_each_owner_lists_and_restores_only_their_folder(tmp_path):
    ds = FakeDataset()
    store = _store(ds)
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


def test_a_conflict_during_a_mirror_is_retried(tmp_path):
    ds = FakeDataset(conflicts=1)
    pauses = []
    store = _store(ds, sleep=pauses.append)
    root = str(tmp_path)
    _conversation(root, "T-C", "Retaining wall", _CREATED)
    up = store.mirror_conversation("T-C", root=root)
    assert not up["errors"] and up["uploaded"] == 4, up
    assert pauses == [0.5] and ds.count("commit") == 1


def test_a_failed_save_never_breaks_the_turn_and_is_sent_next_time(tmp_path):
    ds = FakeDataset(conflicts=99)
    pauses = []
    store = _store(ds, sleep=pauses.append)
    root = str(tmp_path)
    _conversation(root, "T-F", "Bearing", _CREATED)
    up = store.mirror_conversation("T-F", root=root)    # never raises
    assert up["uploaded"] == 0 and up["errors"]
    assert "OpenTransactionAlreadyExists" in up["errors"][0]
    assert len(pauses) == fds.OPEN_TRIES - 1             # once, not per file
    assert ds.files == {}
    ds.conflicts = 0                                     # the other save ends
    up2 = store.mirror_conversation("T-F", root=root)
    assert up2["uploaded"] == 4 and not up2["errors"], up2


def test_a_failed_commit_records_nothing_as_saved(tmp_path):
    ds = FakeDataset()
    ds.fail_commit = True
    store = _store(ds)
    root = str(tmp_path)
    _conversation(root, "T-X", "Settlement", _CREATED)
    up = store.mirror_conversation("T-X", root=root)
    assert up["uploaded"] == 0
    assert any("not saved" in e for e in up["errors"]), up
    assert ds.count("abort") == 1 and ds.files == {} and ds.open is None
    ds.fail_commit = False
    up2 = store.mirror_conversation("T-X", root=root)
    assert up2["uploaded"] == 4 and not up2["errors"], up2


# ---------------------------------------------------------------------------
# configuration: Foundry selects the dataset; everywhere else is unchanged
# ---------------------------------------------------------------------------

class _FakeSDK:
    """A stand-in ``foundry_sdk`` module recording how the client is built."""

    def __init__(self, ds):
        self.built = []
        sdk = types.ModuleType("foundry_sdk")
        sdk.UserTokenAuth = lambda token: ("auth", token)
        sdk.Config = lambda **kw: ("config", kw)

        def client(**kw):
            self.built.append(kw)
            return fake_client(ds)
        sdk.FoundryClient = client
        self.module = sdk


def _foundry(monkeypatch, ds=None, token="made-up-token"):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv(fds.ENV_DATASET, RID)
    if token:
        monkeypatch.setenv("FOUNDRY_TOKEN", token)
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.example.test")
    fake = _FakeSDK(ds or FakeDataset())
    monkeypatch.setitem(sys.modules, "foundry_sdk", fake.module)
    return fake


def test_on_foundry_the_dataset_is_the_permanent_storage(monkeypatch,
                                                         tmp_path):
    ds = FakeDataset()
    fake = _foundry(monkeypatch, ds)
    monkeypatch.setenv(sp.ENV_SITE, "https://example.sharepoint.test/x")
    monkeypatch.setenv(sp.ENV_TOKEN, "made-up")
    store = sp.SharePointStore()
    assert store.configured and store.backend == "foundry"
    assert store.label == "Foundry storage"
    # the agent's SharePoint tools never run on the dataset
    assert sp.configured() is False
    fm = store.file_manager()
    assert isinstance(fm, fds.DatasetFileManager)
    assert fm.dataset == RID and fm.branch == "master"
    root = str(tmp_path)
    _conversation(root, "T-E2E", "Lateral pile", _CREATED)
    assert not store.mirror_conversation("T-E2E", root=root)["errors"]
    assert fake.built[0]["auth"] == ("auth", "made-up-token")
    assert fake.built[0]["hostname"] == "https://stack.example.test"
    assert any(p.startswith("GeotechStaffEngineer/conversations/")
               for p in ds.files)


def test_the_branch_setting(monkeypatch):
    _foundry(monkeypatch)
    monkeypatch.setenv(fds.ENV_BRANCH, "made-up-branch")
    assert fds.DatasetFileManager().branch == "made-up-branch"


def test_the_host_follows_the_apps_settings(monkeypatch):
    fake = _foundry(monkeypatch)
    fds.DatasetFileManager()._client()
    assert fake.built[-1]["hostname"] == "https://stack.example.test"
    assert fds.sdk_target()[1] == "FOUNDRY_HOSTNAME"
    # in a container the SDK finds the host itself
    monkeypatch.setenv(fds.SERVICE_DISCOVERY_ENV, "/made/up/discovery.yml")
    fds.DatasetFileManager()._client()
    assert "hostname" not in fake.built[-1]
    # an explicit GEOTECH_FOUNDRY_HOST wins, its http:// scheme kept
    monkeypatch.setenv("GEOTECH_FOUNDRY_HOST", "http://localhost:9999")
    fds.DatasetFileManager()._client()
    assert fake.built[-1]["hostname"] == "http://localhost:9999"
    assert fake.built[-1]["config"][1]["scheme"] == "http"


def test_a_rotated_token_rebuilds_the_client(monkeypatch):
    fake = _foundry(monkeypatch)
    fm = fds.DatasetFileManager()
    fm._client()
    fm._client()
    assert len(fake.built) == 1
    monkeypatch.setenv("FOUNDRY_TOKEN", "made-up-token-2")
    fm._client()
    assert len(fake.built) == 2
    assert fake.built[-1]["auth"] == ("auth", "made-up-token-2")


def test_sdk_missing_turns_storage_off_with_a_plain_reason(monkeypatch,
                                                         tmp_path):
    _foundry(monkeypatch)
    monkeypatch.setitem(sys.modules, "foundry_sdk", None)   # import fails
    assert not fds.sdk_installed()
    store = sp.SharePointStore()
    assert not store.configured
    assert store.backend == "foundry" and store.label == "Foundry storage"
    reason = fds.unconfigured_reason()
    assert "foundry-platform-sdk" in reason and "not installed" in reason
    root = str(tmp_path)
    _conversation(root, "T-NOSDK", "Pavement", _CREATED)
    up = store.mirror_conversation("T-NOSDK", root=root)    # never raises
    assert up["errors"] and up["errors"][0].startswith(
        "Foundry storage client:")
    assert store.list_remote_conversations() == []
    res = store.restore_conversation("anything_2026-01-02", root=root)
    assert res["status"] == "error" and "not configured" in res["errors"][0]


def test_no_token_is_a_plain_reason_too(monkeypatch):
    _foundry(monkeypatch, token=None)
    assert "no Foundry token" in fds.unconfigured_reason()
    assert not sp.SharePointStore().configured


def test_funhouse_and_tiny_apps_are_unchanged(monkeypatch):
    # the dataset setting alone, off Foundry, changes nothing
    monkeypatch.setenv(fds.ENV_DATASET, RID)
    for deployment in (None, "tinyapps"):
        if deployment:
            monkeypatch.setenv("GEOTECH_DEPLOYMENT", deployment)
        assert not fds.selected()
        store = sp.SharePointStore()
        assert not store.configured and not sp.configured()
        assert store.backend == "sharepoint" and store.label == "SharePoint"
        assert store.root() == sp.DEFAULT_ROOT
        monkeypatch.setenv(sp.ENV_SITE, "https://example.sharepoint.test/x")
        monkeypatch.setenv(sp.ENV_TOKEN, "made-up")
        assert sp.configured() and sp.SharePointStore().configured
        assert sp.SharePointStore().backend == "sharepoint"
        monkeypatch.delenv(sp.ENV_SITE)
        monkeypatch.delenv(sp.ENV_TOKEN)
    # a SharePoint file manager has no batch hook and is driven as before
    assert sp._begin_batch(object()) is False


def test_the_sidebar_names_foundry_storage(monkeypatch, tmp_path):
    """The real shell over a stand-in store whose backend is the dataset:
    the permanent-storage block says Foundry storage, never SharePoint."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.engine_config as engine_config
    from webapp import profiles
    from webapp.engine_config import EngineResolution

    class FoundryStore:
        configured = True
        backend = "foundry"
        label = "Foundry storage"
        last_sync = None

        def list_remote_conversations(self, owner=None, page=None):
            return []

        def mirror_conversation(self, thread_id, root=None):
            return {"uploaded": 0, "skipped": 0, "errors": [],
                    "duration_s": 0.0}

    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    monkeypatch.setattr(engine_config, "resolve_engine",
                        lambda *a, **k: EngineResolution(
                            model=object(), source="prompter",
                            model_name="fake-model", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    store = FoundryStore()
    monkeypatch.setattr(sp, "get_store", lambda *a, **k: store)
    monkeypatch.setenv(profiles.PROFILE_ENV, "geotech")
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    captions = [str(c.value) for c in at.sidebar.caption]
    assert any(c.startswith("Foundry storage is on") for c in captions), \
        captions
    assert any("Foundry storage could not be listed" in c for c in captions)
    assert not any("SharePoint" in c for c in captions), captions


# ---------------------------------------------------------------------------
# Connection diagnostics
# ---------------------------------------------------------------------------

def test_the_diagnostics_line(monkeypatch):
    from webapp import diagnostics as diag
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    c = diag._conversation_storage_check()
    assert c["status"] == diag.SKIP and fds.ENV_DATASET in c["detail"]
    assert "lost when it is rebuilt" in c["detail"]

    ds = FakeDataset()
    ds.files = {"GeotechStaffEngineer/conversations/A/meta.json": b"{}"}
    _foundry(monkeypatch, ds)
    c = diag._conversation_storage_check()
    assert c["status"] == diag.PASS, c
    assert RID in c["detail"] and "FOUNDRY_HOSTNAME" in c["detail"]
    assert "made-up-token" not in c["detail"]
    assert ds.count("create") == 0                  # reads, never writes

    monkeypatch.setitem(sys.modules, "foundry_sdk", None)
    c = diag._conversation_storage_check()
    assert c["status"] == diag.FAIL and "foundry-platform-sdk" in c["detail"]


# ---------------------------------------------------------------------------
# report ingest's durable mirror (RI-203) takes the same file manager
# ---------------------------------------------------------------------------

def test_report_ingests_mirror_fits_unchanged(tmp_path):
    from funhouse_agent.run_mirror import Mirror, file_manager_of
    ds = FakeDataset()
    fm = _fm(ds)
    out = tmp_path / "run_out"
    (out / "runs").mkdir(parents=True)
    (out / "summary.json").write_text('{"n": 2}', encoding="utf-8")
    (out / "runs" / "R01.json").write_text('{"ok": true}', encoding="utf-8")
    mirror = Mirror(sharepoint=fm)
    up = mirror.mirror_dir(out, mirror.remote_for(out.name))
    assert up["uploaded"] == 2 and not up["errors"], up
    assert "GeotechStaffEngineer/report_ingest/run_out/runs/R01.json" in ds.files
    back = tmp_path / "wiped"
    res = mirror.restore_dir(mirror.remote_for(out.name), back)
    assert res["downloaded"] == 2 and not res["errors"], res
    assert (back / "runs" / "R01.json").read_text(encoding="utf-8") == \
        '{"ok": true}'
    # the app's store is unwrapped to the same file manager
    assert file_manager_of(sp.SharePointStore(file_manager=fm)) is fm
