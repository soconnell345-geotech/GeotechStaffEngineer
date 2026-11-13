"""TA-40: SharePoint saves off the turn and off the page.

A fake Graph stands in for the service (``urllib.request.urlopen`` is
replaced), as in the other SharePoint tests: one document library that
keeps what is put into it, counts every call, and can be told to hold a
call, fail a piece of a large upload, throttle, or refuse a name. Every
tenant, site, person and file here is made up.

Each test fails on master before TA-40 (its docstring says how).
"""

import hashlib
import io
import json
import os
import re
import threading
import tracemalloc
import urllib.error
import urllib.parse
import urllib.request

import pytest

import aec_document_review.core as core
import aec_document_review.graph_sharepoint as gs
import aec_document_review.sharepoint_store as sp

SITE = "https://contoso.sharepoint.com/sites/Alpha"
WAIT = 10.0                       # seconds a test waits before it gives up
MAX_MB = "GEOTECH_SHAREPOINT_MAX_UPLOAD_MB"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET,
                 sp.ENV_TOKEN, sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT,
                 MAX_MB, gs.ENV_CLOUD, gs.ENV_LOGIN_URL,
                 gs.ENV_API_URL, "GEOTECH_FOUNDRY_CONVERSATION_DATASET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gs, "RESUME_PAUSE_S", 0.0, raising=False)
    yield


# ------------------------------------------------------------ the fake Graph

class _Resp:
    def __init__(self, data):
        self._d = data if isinstance(data, bytes) else json.dumps(data).encode()

    def read(self):
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http(url, code, headers=None, body=None):
    return urllib.error.HTTPError(
        url, code, "err", headers or {},
        io.BytesIO(json.dumps(body or {"error": {"code": "x"}}).encode()))


_REFUSED = set('"*:<>?\\|')


class FakeGraph:
    """One site, one library ("Shared Documents", drive ``D1``)."""

    def __init__(self, keep_bytes=True):
        self.calls = []                 # (method, address without its query)
        self.files = {}                 # path -> bytes, or (size, sha256 hex)
        self.keep_bytes = keep_bytes
        self.sessions = {}              # upload address -> its state
        self.biggest_body = 0
        self.refuse_names = False       # answer 400 to a refused name
        self.before = None              # hook(method, path, req): may raise
        self._ids = {}
        self._guard = threading.Lock()

    # -- counting -----------------------------------------------------------
    def count(self, method, pattern):
        return len([1 for m, u in self.calls
                    if m == method and re.search(pattern, u)])

    def _id(self, path):
        with self._guard:
            for k, v in self._ids.items():
                if v == path:
                    return k
            key = f"I{len(self._ids) + 1}"
            self._ids[key] = path
            return key

    def _is_folder(self, path):
        return any(p.startswith(path + "/") for p in list(self.files))

    def _store(self, path, data):
        self.files[path] = data

    # -- the service --------------------------------------------------------
    def __call__(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        shown = url.split("?", 1)[0]
        with self._guard:
            self.calls.append((method, shown))
        if req.data:
            self.biggest_body = max(self.biggest_body, len(req.data))
        if "/oauth2/v2.0/token" in url:
            return _Resp({"access_token": "TOKEN", "expires_in": 3600})
        if shown in self.sessions:
            return self._session(req, shown)
        path = urllib.parse.unquote(shown.split("/v1.0/", 1)[1])
        if self.before is not None:
            self.before(method, path, req)
        if re.match(r"sites/[^:/]+:/sites/[^/]+$", path):
            return _Resp({"id": "SA"})
        if path == "sites/SA/drives":
            return _Resp({"value": [{
                "id": "D1", "name": "Documents",
                "webUrl": f"{SITE}/Shared%20Documents"}]})
        if path == "drives/D1/root" and method == "GET":
            return _Resp({"id": self._id(""), "folder": {}})
        m = re.match(r"drives/D1/items/([^/]+)/children$", path)
        if m and method == "GET":
            base = self._ids[m.group(1)]
            head = base + "/" if base else ""
            names = {}
            for p in list(self.files):
                if p.startswith(head):
                    first, _, more = p[len(head):].partition("/")
                    names[first] = bool(more) or names.get(first, False)
            return _Resp({"value": [
                dict({"id": self._id(head + n), "name": n, "size": 1},
                     **({"folder": {}} if is_dir else {"file": {}}))
                for n, is_dir in sorted(names.items())]})
        if m and method == "POST":
            name = json.loads(req.data)["name"]
            return _Resp({"id": self._id(name), "name": name, "folder": {}})
        m = re.match(r"drives/D1/root:/(.+):/createUploadSession$", path)
        if m and method == "POST":
            self._check_name(url, m.group(1))
            address = f"https://upload.example/session/{len(self.sessions)}"
            self.sessions[address] = {
                "path": m.group(1), "got": 0, "parts": [],
                "sha": hashlib.sha256(), "fail": []}
            return _Resp({"uploadUrl": address + "?tempauth=SECRET"})
        m = re.match(r"drives/D1/root:/(.+):/content$", path)
        if m and method == "PUT":
            self._check_name(url, m.group(1))
            if "conflictBehavior=fail" in url and m.group(1) in self.files:
                raise _http(url, 409)
            self._store(m.group(1), req.data)
            return _Resp({"id": self._id(m.group(1))})
        if m and method == "GET":
            if m.group(1) not in self.files:
                raise _http(url, 404)
            return _Resp(self.files[m.group(1)])
        m = re.match(r"drives/D1/root:/(.+)$", path)
        if m and method == "GET":
            inner = m.group(1)
            if inner in self.files:
                return _Resp({"id": self._id(inner), "file": {},
                              "name": inner.rsplit("/", 1)[-1]})
            if self._is_folder(inner):
                return _Resp({"id": self._id(inner), "folder": {},
                              "name": inner.rsplit("/", 1)[-1]})
            raise _http(url, 404)
        raise AssertionError(f"unexpected call {method} {url}")

    def _check_name(self, url, path):
        if not self.refuse_names:
            return
        for part in path.split("/"):
            if set(part) & _REFUSED or part != part.strip(" ") \
                    or part.endswith("."):
                raise _http(url, 400, body={"error": {
                    "code": "invalidRequest",
                    "message": "The provided name cannot contain any "
                               "illegal characters."}})

    def _session(self, req, shown):
        state = self.sessions[shown]
        if req.get_method() == "GET":
            return _Resp({"nextExpectedRanges": [f"{state['got']}-"]})
        rng = req.headers.get("Content-range")
        start, end, total = map(int, re.match(
            r"bytes (\d+)-(\d+)/(\d+)", rng).groups())
        state["parts"].append((start, end))
        if state["fail"]:
            what = state["fail"].pop(0)
            if what == "lost after half":      # half arrived, then the line
                half = len(req.data) // 2      # dropped
                state["sha"].update(req.data[:half])
                if self.keep_bytes:
                    state.setdefault("data", bytearray()).extend(
                        req.data[:half])
                state["got"] += half
                raise urllib.error.URLError("connection reset")
            if what == "error":
                raise _http(shown, 500)
        if start != state["got"]:
            raise _http(shown, 416)
        state["sha"].update(req.data)
        if self.keep_bytes:
            state.setdefault("data", bytearray()).extend(req.data)
        state["got"] = end + 1
        if state["got"] == total:
            self._store(state["path"], bytes(state["data"])
                        if self.keep_bytes
                        else (total, state["sha"].hexdigest()))
            return _Resp({"id": "DONE"})
        return _Resp({"nextExpectedRanges": [f"{state['got']}-"]})


@pytest.fixture
def graph(monkeypatch):
    fake = FakeGraph()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def _store():
    fm = gs.GraphFileManager(site_url=SITE, tenant_id="made-up-tenant",
                             client_id="made-up-client",
                             client_secret="made-up-secret")
    return sp.SharePointStore(file_manager=fm)


def _conversation(root, tid, files=None, title=None, owner=None):
    core.ensure_conversation(tid, root=root)
    conv = core.conversation_dir(tid, root)
    if title or owner:
        meta = core.load_meta(tid, root) or {"thread_id": tid}
        if title:
            meta["title"] = title
        if owner:
            meta["owner"] = owner
        core.save_meta(tid, meta, root)
    for rel, data in (files or {"files/calc.pdf": b"%PDF made up"}).items():
        local = os.path.join(conv, *rel.split("/"))
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as fh:
            fh.write(data)
    return conv


def _in_thread(fn, *args):
    out = {}

    def run():
        out["result"] = fn(*args)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, out


# ----------------------------------- 1. two conversations, never one queue

def test_two_conversations_mirror_at_once(tmp_path, graph):
    """One conversation's upload is held at the service; another
    conversation's mirror runs to its end meanwhile. On master the second
    waits behind the store's one lock until the first is let go."""
    root = str(tmp_path)
    _conversation(root, "AAAA1111", {"files/big-set.pdf": b"a" * 100})
    _conversation(root, "BBBB2222", {"files/note.pdf": b"b" * 100})
    held, release = threading.Event(), threading.Event()

    def before(method, path, req):
        if method == "PUT" and "AAAA1111" in path and "big-set" in path:
            held.set()
            assert release.wait(WAIT)

    graph.before = before
    store = _store()
    first, first_out = _in_thread(store.mirror_conversation, "AAAA1111", root)
    try:
        assert held.wait(WAIT)
        second, second_out = _in_thread(store.mirror_conversation,
                                        "BBBB2222", root)
        second.join(WAIT)
        assert not second.is_alive(), \
            "the second conversation's mirror waited on the first's"
        assert second_out["result"]["errors"] == []
        assert second_out["result"]["uploaded"] >= 1
    finally:
        release.set()
        first.join(WAIT)
    assert first_out["result"]["errors"] == []


def test_save_after_a_turn_is_queued_not_run_in_the_turn(tmp_path, graph):
    """On the Graph route ``save_after_change`` returns at once (the save is
    queued, as on the Foundry dataset) and two conversations' queued saves
    run at the same time. On master it mirrors inside the call: the call
    does not come back while the upload is held."""
    root = str(tmp_path)
    for tid in ("AAAA1111", "BBBB2222"):
        _conversation(root, tid, {"files/a.pdf": b"x" * 10})
        core.register_thread_root(tid, root)
    held, release = threading.Event(), threading.Event()

    def before(method, path, req):
        if method == "PUT" and "AAAA1111" in path and "a.pdf" in path:
            held.set()
            assert release.wait(WAIT)

    graph.before = before
    store = _store()
    assert store.background is True
    call, out = _in_thread(sp.save_after_change, store, "AAAA1111")
    try:
        call.join(WAIT / 2)
        assert not call.is_alive(), "the turn waited for its save"
        assert out["result"] is None
        assert held.wait(WAIT)
        assert store.save_state("AAAA1111") == "saving"
        assert sp.save_after_change(store, "BBBB2222") is None
        # the second conversation's save ends while the first is held
        deadline = threading.Event()
        for _ in range(int(WAIT * 20)):
            if store.last_sync_for("BBBB2222") is not None:
                break
            deadline.wait(0.05)
        done = store.last_sync_for("BBBB2222")
        assert done is not None and done["errors"] == []
        assert store.last_sync_for("AAAA1111") is None
    finally:
        release.set()
    assert store.flush(WAIT)
    assert store.last_sync_for("AAAA1111")["errors"] == []
    assert store.save_state("AAAA1111") is None


def test_a_throttled_wait_holds_only_its_own_conversation(tmp_path, graph,
                                                          monkeypatch):
    """The service throttles one conversation's upload (429, Retry-After);
    while that mirror waits, another conversation's mirror is sent and
    ends. On master the wait is taken under the store's one lock."""
    root = str(tmp_path)
    _conversation(root, "AAAA1111", {"files/a.pdf": b"a"})
    _conversation(root, "BBBB2222", {"files/b.pdf": b"b"})
    waiting, release = threading.Event(), threading.Event()
    throttled = []

    def before(method, path, req):
        if method == "PUT" and "AAAA1111" in path and not throttled:
            throttled.append(path)
            raise _http(req.full_url, 429, headers={"Retry-After": "5"})

    def wait(seconds):                       # the wait the service named
        waiting.set()
        assert release.wait(WAIT)

    graph.before = before
    monkeypatch.setattr(gs.time, "sleep", wait)
    store = _store()
    first, first_out = _in_thread(store.mirror_conversation, "AAAA1111", root)
    try:
        assert waiting.wait(WAIT)
        second, second_out = _in_thread(store.mirror_conversation,
                                        "BBBB2222", root)
        second.join(WAIT)
        assert not second.is_alive(), \
            "another conversation waited out this one's throttle"
        assert second_out["result"]["errors"] == []
    finally:
        release.set()
        first.join(WAIT)
    assert first_out["result"]["errors"] == []       # sent after the wait


def test_two_new_conversations_of_one_title_get_two_folders(tmp_path, graph):
    """Two conversations with the same title, first mirrored at the same
    moment: while the first is still checking the library for its new
    folder name, the second takes another name and is mirrored; they end
    in two folders. On master the second waits behind the first."""
    root = str(tmp_path)
    _conversation(root, "AAAA1111", title="Sheet A-101 review")
    _conversation(root, "BBBB2222", title="Sheet A-101 review")
    checking, release = threading.Event(), threading.Event()

    def before(method, path, req):
        if method == "GET" and path.endswith("/conversations") \
                and not checking.is_set():
            checking.set()
            assert release.wait(WAIT)

    graph.before = before
    store = _store()
    a, a_out = _in_thread(store.mirror_conversation, "AAAA1111", root)
    try:
        assert checking.wait(WAIT)
        b, b_out = _in_thread(store.mirror_conversation, "BBBB2222", root)
        b.join(WAIT)
        assert not b.is_alive(), "the second mirror waited on the first's"
    finally:
        release.set()
        a.join(WAIT)
    assert a_out["result"]["errors"] == [] == b_out["result"]["errors"]
    folders = {a_out["result"]["folder"], b_out["result"]["folder"]}
    assert len(folders) == 2 and None not in folders
    metas = [json.loads(v) for p, v in graph.files.items()
             if p.endswith("/meta.json")]
    assert sorted(m["thread_id"] for m in metas) == ["AAAA1111", "BBBB2222"]


def test_other_file_managers_keep_one_lock_and_save_in_the_turn(tmp_path):
    """A file manager that does not say it serves several mirrors at once
    (the Funhouse SDK's, a test's stand-in) is driven as before: one lock,
    the save made inside ``save_after_change``, its folders made first."""
    class Plain:
        def __init__(self):
            self.folders, self.uploads = [], []

        def create_folder(self, path):
            self.folders.append(path)

        def upload_file(self, local, remote, overwrite=False):
            self.uploads.append(remote)
            return True

        def get_web_url(self, path):
            return f"https://sp.example/{path}"

    root = str(tmp_path)
    _conversation(root, "AAAA1111")
    core.register_thread_root("AAAA1111", root)
    fm = Plain()
    store = sp.SharePointStore(file_manager=fm)
    assert store.per_conversation is False and store.background is False
    assert store._lock_for("AAAA1111") is store._lock_for("BBBB2222")
    summary = sp.save_after_change(store, "AAAA1111")
    assert summary["uploaded"] >= 1 and summary["errors"] == []
    assert fm.folders, "the folders were not made first"


# --------------------------------------------- 2. no folder walk per upload

def test_first_mirror_of_40_files_walks_no_folders(tmp_path, graph):
    """A 40-file first mirror: one request per file, no folder made or
    listed for any of them (the one listing left is the check that the
    conversation's new folder name is free). On master each upload looks up
    every level of its path: several hundred calls."""
    root = str(tmp_path)
    files = {f"files/set/sheet-{i:02d}.pdf": b"%PDF " + bytes([65 + i % 20])
             for i in range(36)}
    files.update({f"files/out-{i}.docx": b"docx" for i in range(4)})
    _conversation(root, "AAAA1111", files, title="Tower set review")
    store = _store()
    summary = store.mirror_conversation("AAAA1111", root)
    assert summary["errors"] == []
    assert summary["uploaded"] >= 40
    assert graph.count("POST", r"/children$") == 0        # no folder made
    assert graph.count("GET", r"/children$") <= 1         # the name check
    puts = graph.count("PUT", r":/content$")
    assert puts == summary["uploaded"]
    assert all("/root:/" in u for m, u in graph.calls
               if m == "PUT")                             # by path
    assert len(graph.calls) <= puts + 8
    assert any(p.endswith("/files/set/sheet-07.pdf") for p in graph.files)


def test_the_upload_tool_makes_no_folders_first(tmp_path, graph):
    """``GraphFileManager.upload_file`` to a folder that does not exist yet:
    one request for the file (after sign-in and the library look-up). On
    master the folder chain is walked and made first."""
    local = tmp_path / "memo.docx"
    local.write_bytes(b"made-up memo")
    fm = _store().file_manager()
    assert fm.MAKES_FOLDERS is True
    assert fm.upload_file(str(local),
                          "Shared Documents/Team/2026/out/memo.docx") is True
    assert graph.files["Team/2026/out/memo.docx"] == b"made-up memo"
    assert graph.count("POST", r"/children$") == 0
    assert graph.count("GET", r"/children$") == 0
    # the name is taken: False, as the callers' new-name retry reads it
    assert fm.upload_file(str(local),
                          "Shared Documents/Team/2026/out/memo.docx") is False


# ------------------------------------- 3. a failed piece is taken up again

def _small_pieces(monkeypatch):
    monkeypatch.setattr(gs, "SIMPLE_UPLOAD_MAX", 64 * 1024)
    monkeypatch.setattr(gs, "CHUNK_SIZE", 320 * 1024)


def test_a_failed_piece_is_resumed(tmp_path, graph, monkeypatch):
    """A large upload whose second piece is cut off half way: the session
    is asked where it stands and the upload goes on from there; the file
    arrives whole. On master the upload fails with the piece."""
    _small_pieces(monkeypatch)
    data = os.urandom(1000 * 1024)
    local = tmp_path / "survey.dwg"
    local.write_bytes(data)
    fm = _store().file_manager()
    real = graph.__call__

    def fail_second_piece(req, timeout=None):
        if "/createUploadSession" in req.full_url:
            resp = real(req, timeout)
            next(iter(graph.sessions.values()))["fail"] = []
            return resp
        return real(req, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", fail_second_piece)
    # arm the failure once the first piece is in
    original_session = graph._session

    def session(req, shown):
        state = graph.sessions[shown]
        if req.get_method() == "PUT" and len(state["parts"]) == 1 \
                and not state.get("armed"):
            state["armed"] = True
            state["fail"] = ["lost after half"]
        return original_session(req, shown)

    graph._session = session
    assert fm.upload_file(str(local), "Shared Documents/GSE/survey.dwg",
                          overwrite=True) is True
    assert graph.files["GSE/survey.dwg"] == data
    state = next(iter(graph.sessions.values()))
    assert graph.count("GET", r"upload\.example/session") == 1
    # the piece after the failure starts where the service said: half way
    # through the failed one, not at its start and not after it
    starts = [s for s, _e in state["parts"]]
    assert starts[2] == 320 * 1024 + 160 * 1024
    assert graph.count("POST", r"root:/GSE/survey\.dwg:/createUploadSession") == 1


def test_a_piece_that_keeps_failing_ends_the_upload(tmp_path, graph,
                                                   monkeypatch):
    """The resumes are bounded: after ``RESUME_TRIES`` the upload fails
    with the piece's own error (and names no token)."""
    _small_pieces(monkeypatch)
    local = tmp_path / "survey.dwg"
    local.write_bytes(os.urandom(700 * 1024))
    fm = _store().file_manager()
    original_session = graph._session

    def session(req, shown):
        if req.get_method() == "PUT":
            graph.sessions[shown]["fail"] = ["error"]
        return original_session(req, shown)

    graph._session = session
    with pytest.raises(gs.GraphError) as err:
        fm.upload_file(str(local), "Shared Documents/GSE/survey.dwg",
                       overwrite=True)
    assert err.value.status == 500
    assert "SECRET" not in str(err.value)
    assert graph.count("PUT", r"upload\.example/session") == gs.RESUME_TRIES + 1
    assert "GSE/survey.dwg" not in graph.files


# ------------------------------------ 4. large files: in pieces, with a limit

def test_a_large_file_goes_from_disk_in_pieces(tmp_path, monkeypatch):
    """A 24 MB file is sent in pieces read from disk one at a time: the
    memory in use never comes near the file's size. On master the whole
    file is read into memory first."""
    monkeypatch.setattr(gs, "CHUNK_SIZE", 320 * 1024)
    fake = FakeGraph(keep_bytes=False)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    size = 24 * 1024 * 1024
    local = tmp_path / "site-model.rvt"
    sha = hashlib.sha256()
    with open(local, "wb") as fh:
        block = os.urandom(1024 * 1024)
        for _ in range(24):
            fh.write(block)
            sha.update(block)
    del block
    fm = _store().file_manager()
    tracemalloc.start()
    try:
        assert fm.upload_file(str(local), "Shared Documents/GSE/site-model.rvt",
                              overwrite=True) is True
        _now, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert fake.files["GSE/site-model.rvt"] == (size, sha.hexdigest())
    assert fake.biggest_body <= 320 * 1024
    assert peak < size / 4, f"{peak / 1e6:.1f} MB held for a 24 MB file"
    assert fake.count("POST", r"root:/GSE/site-model\.rvt:/createUploadSession") == 1


def test_a_file_above_the_limit_is_named_once_and_skipped(tmp_path, graph,
                                                          monkeypatch):
    """A file above the size limit is not sent. The first mirror says so as
    an error that names the file and the limit; later mirrors do not try it
    or fail it again, and still name it in the summary's note; the other
    files go up. On master there is no limit: the file is sent."""
    monkeypatch.setenv(MAX_MB, "1")
    root = str(tmp_path)
    conv = _conversation(root, "AAAA1111", {
        "files/point-cloud.las": b"p" * (2 * 1024 * 1024),
        "files/memo.docx": b"memo"})
    store = _store()
    first = store.mirror_conversation("AAAA1111", root)
    assert len(first["errors"]) == 1
    assert "point-cloud.las" in first["errors"][0]
    assert "1 MB limit" in first["errors"][0]
    assert not any("point-cloud" in p for p in graph.files)
    assert any(p.endswith("/files/memo.docx") for p in graph.files)
    calls = len(graph.calls)

    second = store.mirror_conversation("AAAA1111", root)
    assert second["errors"] == []
    assert second["not_sent"] == ["files/point-cloud.las"]
    assert "point-cloud.las" in second["note"] and "1 MB" in second["note"]
    assert not any(m == "PUT" for m, _u in graph.calls[calls:])

    # the file changes (and now fits): it is sent, and the note is gone
    with open(os.path.join(conv, "files", "point-cloud.las"), "wb") as fh:
        fh.write(b"small now")
    third = store.mirror_conversation("AAAA1111", root)
    assert third["errors"] == [] and "note" not in third
    assert any(p.endswith("/files/point-cloud.las") for p in graph.files)


# ------------------------------------------- 5. a listing that failed says so

def test_a_failed_listing_reads_as_a_failure(tmp_path, graph):
    """The library cannot be read (HTTP 500): the listing is empty AND says
    why, and the store keeps the reason until a listing succeeds. A library
    with no conversations folder yet is empty with no error. On master both
    come back as the same bare empty list."""
    store = _store()
    empty = store.list_remote_conversations()
    assert list(empty) == [] and empty.error is None
    assert store.listing_error() is None

    def before(method, path, req):
        if method == "GET" and path.endswith("/conversations"):
            raise _http(req.full_url, 500)

    graph.before = before
    failed = store.list_remote_conversations()
    assert list(failed) == []
    assert failed.error and "could not be listed" in failed.error
    assert "500" in failed.error
    assert store.listing_error() == failed.error
    # one person's failure is not another's, nor another page's
    assert store.listing_error(owner="corp_j.doe") is None

    graph.before = None
    root = str(tmp_path)
    _conversation(root, "AAAA1111", title="Sheet A-101 review")
    assert store.mirror_conversation("AAAA1111", root)["errors"] == []
    listed = store.list_remote_conversations()
    assert listed.error is None and store.listing_error() is None
    assert [e["name"] for e in listed] == [
        os.path.basename(store.session_folder("AAAA1111", root))]


# ---------------------------------- 6. a name SharePoint refuses is mapped

@pytest.mark.parametrize("name", [
    'plan: rev 2.pdf', 'what?.txt', 'a*b.csv', 'x<y>.md', 'q|r.txt',
    'he said "no".docx', 'back\\slash.txt', 'trailing dot.', 'trailing ',
    ' leading.txt', 'CON', 'con.txt', 'LPT3.log', 'desktop.ini', '.lock',
    '~$draft.docx', 'a_vti_b.txt'])
def test_a_refused_name_maps_to_an_accepted_one(name):
    """Every refused form maps to a name the service accepts, the same one
    each time, keeping the extension; an accepted name is left alone."""
    kept = sp.remote_name(name)
    assert kept != name and kept == sp.remote_name(name)
    assert not set(kept) & set('"*:<>?/\\|')
    assert kept == kept.strip(" ") and not kept.endswith(".")
    assert kept.lower() not in sp._RESERVED_NAMES
    assert kept.split(".", 1)[0].lower() not in sp._RESERVED_NAMES
    assert not kept.startswith("~$") and "_vti_" not in kept.lower()
    assert sp.remote_name(kept) == kept
    ext = os.path.splitext(name.strip(" ").rstrip(". "))[1]
    if ext and name not in ('desktop.ini', '.lock'):
        assert kept.endswith(ext)


def test_accepted_names_are_left_alone():
    for name in ("calc.pdf", "Sheet A-101 (rev 2).pdf", "50% set #3.dwg",
                 "meta.json", "résumé.docx", ".hidden", "console.log"):
        assert sp.remote_name(name) == name
    assert sp.remote_rel("files/set/a.pdf") == "files/set/a.pdf"
    assert sp.remote_name("a:b.pdf") != sp.remote_name("a?b.pdf")


@pytest.mark.skipif(os.name == "nt",
                    reason="Windows cannot hold such a file name")
def test_a_refused_name_does_not_fail_twice(tmp_path, graph):
    """A file whose name SharePoint refuses (a colon): it goes up under an
    accepted name at the first mirror, with a note, and is up to date at
    the second; a restore on another host gives it its own name back. On
    master it is refused, the same way, at every sync."""
    graph.refuse_names = True
    root = str(tmp_path / "host1")
    conv = _conversation(root, "AAAA1111", {
        "files/plan: rev 2.pdf": b"%PDF the plan",
        "files/memo.docx": b"memo"}, title="Sheet A-101 review")
    store = _store()
    first = store.mirror_conversation("AAAA1111", root)
    assert first["errors"] == []
    kept = sp.remote_name("plan: rev 2.pdf")
    assert any(p.endswith(f"/files/{kept}") for p in graph.files)
    assert "plan: rev 2.pdf" in first["note"] and kept in first["note"]
    assert sp.mirrored_rel(conv, "files/plan: rev 2.pdf") == f"files/{kept}"
    assert sp.mirrored_rel(conv, "files/memo.docx") == "files/memo.docx"
    # a manifest an older version still reads: folder and files as before
    manifest = sp._load_manifest(conv)
    assert set(manifest["files"]) >= {"files/plan: rev 2.pdf",
                                      "files/memo.docx", "meta.json"}
    assert isinstance(manifest["folder"], str)

    calls = len(graph.calls)
    second = store.mirror_conversation("AAAA1111", root)
    assert second["errors"] == [] and second["uploaded"] == 0
    assert "note" not in second
    assert not any(m == "PUT" for m, _u in graph.calls[calls:])

    # another host: the conversation comes back with the file's own name
    other = str(tmp_path / "host2")
    os.makedirs(other)
    restored = _store().restore_conversation(
        os.path.basename(first["folder"]), root=other)
    assert restored["status"] == "restored" and restored["errors"] == []
    back = core.conversation_dir("AAAA1111", other)
    with open(os.path.join(back, "files", "plan: rev 2.pdf"), "rb") as fh:
        assert fh.read() == b"%PDF the plan"
    assert not os.path.exists(os.path.join(back, sp.NAMES_NAME))
    assert not os.path.exists(os.path.join(back, "files", kept))
    # and the next mirror from there sends nothing again
    core.register_thread_root("AAAA1111", other)
    again = _store().mirror_conversation("AAAA1111", other)
    assert again["errors"] == [] and again["uploaded"] == 0


def test_a_conversation_with_plain_names_has_the_manifest_it_always_had(
        tmp_path, graph):
    """No refused name and nothing over the limit: the manifest holds
    ``folder`` and ``files`` and nothing else, and no names file is sent."""
    root = str(tmp_path)
    conv = _conversation(root, "AAAA1111")
    summary = _store().mirror_conversation("AAAA1111", root)
    assert summary["errors"] == [] and "note" not in summary
    assert set(sp._load_manifest(conv)) == {"folder", "files"}
    assert sp.NAMES_NAME not in sp._load_manifest(conv)["files"]
    assert not any(p.endswith(sp.NAMES_NAME) for p in graph.files)
