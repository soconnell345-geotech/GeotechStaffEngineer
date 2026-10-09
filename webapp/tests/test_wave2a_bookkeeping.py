"""Live smoke wave 2a fixes in the web app's conversation bookkeeping.

* B2  -- a mid-turn SharePoint mirror ("save it to SharePoint") never uploads
  the turn's ``partial.json`` checkpoint (or any half-written ``.part`` /
  ``.tmp`` file); a restore never downloads one; and a checkpoint whose turn
  the transcript shows as answered is stale, not an interruption.
* B10 -- the next turn's file note names a chart's PNG beside its
  ``.plotly.json`` sidecar.
* B11 -- the first typed question extends a files title instead of replacing
  it ("3000.pdf — What is this sheet?"); the SharePoint folder never moves.
* B12 -- a new file-named SharePoint folder is checked against the remote
  library, not only this host's conversations.
"""

import json
import os
import time

import pytest

import webapp.core as core
import webapp.sharepoint_store as sp
import webapp.turn_jobs as tj

DASH = "—"


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    for env in (sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET,
                sp.ENV_TOKEN, sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT):
        monkeypatch.delenv(env, raising=False)
    tj._JOBS.clear()
    yield


class RemoteFM:
    """A fake SharePoint file manager over an in-memory tree
    ``{remote_path: bytes}``: upload, ls, download and delete."""

    def __init__(self):
        self.tree = {}
        self.uploads = []
        self.deleted = []
        self.ls_calls = []

    def create_folder(self, path):
        return True

    def upload_file(self, local, remote, overwrite=False):
        with open(local, "rb") as fh:
            self.tree[remote] = fh.read()
        self.uploads.append(remote)
        return True

    def get_web_url(self, path):
        return f"https://sp.example/{path}"

    def put(self, remote, data):
        self.tree[remote] = data if isinstance(data, bytes) else \
            json.dumps(data).encode("utf-8")

    def ls(self, path):
        self.ls_calls.append(path)
        path = path.rstrip("/")
        seen = {}
        for rp in self.tree:
            if not rp.startswith(path + "/"):
                continue
            head, _, tail = rp[len(path) + 1:].partition("/")
            if tail:
                seen.setdefault(head, {"name": head, "type": "folder",
                                       "path": f"{path}/{head}"})
            else:
                seen[head] = {"name": head, "type": "file", "path": rp}
        return list(seen.values())

    def download_file(self, path, local_path=None, return_bytes=True,
                      overwrite=False):
        if path not in self.tree:
            raise FileNotFoundError(path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(self.tree[path])
        return True

    def delete_file(self, path):
        self.deleted.append(path)
        self.tree.pop(path, None)
        return True


def _ctx(files, transcript, prompt, **extra):
    ctx = {"prompt": prompt, "temp_dir": files,
           "before": core.snapshot_dir(files),
           "before_mtimes": core.snapshot_mtimes(files),
           "staged_inputs": set(), "working_dir": files, "before_wd": None,
           "artifacts": [], "artifacts_before_len": 0,
           "transcript": transcript, "trace_on": False, "model": "m",
           "behavior": core.default_behavior()}
    ctx.update(extra)
    return ctx


def _run_turn(monkeypatch, tid, files, transcript, prompt, during=None,
              **extra):
    """One turn through the real worker, the way app.py starts it: the
    question is written to the transcript, then the checkpoint."""
    def stream_turn(agent, messages, thread_id, recursion_limit=None, **kw):
        if during:
            during()
        yield {"kind": "turn_done", "answer": f"answer to {prompt}",
               "turn_tokens": 1}
    monkeypatch.setattr(core, "stream_turn", stream_turn)
    entry = {"role": "user", "text": prompt}
    transcript.append(entry)
    core.append_transcript(tid, entry)
    core.begin_partial(tid, prompt)
    job = tj.start_turn_job(object(), [{"role": "user", "content": prompt}],
                            tid, 50, _ctx(files, transcript, prompt, **extra))
    t0 = time.time()
    while not job.done and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.done
    return job


# ---------------------------------------------------------------------------
# B2: in-flight files stay off the mirror, and a stale checkpoint is not an
# interruption
# ---------------------------------------------------------------------------

def test_which_files_are_in_flight():
    assert core.is_in_flight_file("partial.json")
    assert core.is_in_flight_file("downloads.json.tmp")
    assert core.is_in_flight_file("files/.sp_download_ab12.part")
    assert core.is_in_flight_file("files\\coverage.json.part")
    assert not core.is_in_flight_file("files/partial.json")   # the user's
    assert not core.is_in_flight_file("transcript.jsonl")
    assert not core.is_in_flight_file("files/memo.docx")
    assert not core.is_in_flight_file("")


def test_a_mid_turn_mirror_leaves_the_checkpoint_out_and_restore_adds_no_fake_turn(
        monkeypatch, tmp_path):
    """F04: "save it to SharePoint" mirrored the conversation in the middle
    of a turn and uploaded partial.json; restoring the conversation then
    appended "this turn was interrupted"."""
    fm = RemoteFM()
    store = sp.SharePointStore(file_manager=fm)
    tid = "B2MIDTURN001"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    transcript = []
    _run_turn(monkeypatch, tid, files, transcript, "first question")
    core.rename_conversation(tid, "21.01.pdf")

    seen = {}

    def save_to_sharepoint():          # the agent's mid-turn mirror
        with open(os.path.join(files, "memo.docx"), "wb") as fh:
            fh.write(b"memo")
        # a SharePoint download half done in parallel
        with open(os.path.join(files, ".sp_download_x.part"), "wb") as fh:
            fh.write(b"half")
        seen["partial_on_disk"] = os.path.isfile(core.partial_path(tid))
        seen["summary"] = store.mirror_conversation(tid)
        os.remove(os.path.join(files, ".sp_download_x.part"))
    _run_turn(monkeypatch, tid, files, transcript, "save the memo to SharePoint",
              during=save_to_sharepoint)
    assert seen["partial_on_disk"]                 # the checkpoint was live
    assert not seen["summary"]["errors"]
    names = [r.rsplit("/", 1)[-1] for r in fm.uploads]
    assert "memo.docx" in names
    assert "partial.json" not in names
    assert not any(n.endswith(".part") for n in names)

    # the end-of-turn mirror, then a wiped host restores the conversation
    store.mirror_conversation(tid)
    folder = core.load_meta(tid)[sp.MIRROR_FOLDER_KEY]
    other = str(tmp_path / "after_restart")
    res = store.restore_conversation(folder, root=other)
    assert res["status"] == "restored", res
    assert not os.path.isfile(core.partial_path(tid, other))
    assert core.recover_partial(tid, other) is None
    roles = [e["role"] for e in core.load_transcript(tid, other)]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_a_checkpoint_uploaded_by_an_older_version_is_not_restored_and_is_removed(
        tmp_path):
    root = str(tmp_path / "host")
    tid = "B2LEGACY0001"
    core.ensure_conversation(tid, title="Bridge plans", root=root)
    core.append_transcript(tid, {"role": "user", "text": "q1"}, root=root)
    core.append_transcript(tid, {"role": "assistant", "text": "a1"}, root=root)
    fm = RemoteFM()
    store = sp.SharePointStore(file_manager=fm)
    store.mirror_conversation(tid, root=root)
    folder = store.session_folder(tid, root=root)
    # what an older version left in the folder, and in its manifest
    fm.put(f"{folder}/partial.json",
           {"prompt": "q1", "text": "", "updated": 1.0})
    conv = core.conversation_dir(tid, root)
    manifest = sp._load_manifest(conv)
    manifest["files"]["partial.json"] = [115, 1.0]
    sp._save_manifest(conv, manifest["files"], manifest["folder"])

    # restore on another host: the checkpoint is not downloaded
    other = str(tmp_path / "other")
    res = store.restore_conversation(folder.rsplit("/", 1)[-1], root=other)
    assert res["status"] == "restored", res
    assert not os.path.isfile(core.partial_path(tid, other))
    assert core.recover_partial(tid, other) is None

    # the next mirror from the first host deletes it and forgets it
    store.mirror_conversation(tid, root=root)
    assert f"{folder}/partial.json" in fm.deleted
    assert f"{folder}/partial.json" not in fm.tree
    assert "partial.json" not in sp._manifest_files(sp._load_manifest(conv))


def test_a_stale_checkpoint_is_not_an_interruption(tmp_path):
    root, tid = str(tmp_path), "B2STALE"
    for role, text in (("user", "q1"), ("assistant", "a1"),
                       ("user", "q2"), ("assistant", "a2")):
        core.append_transcript(tid, {"role": role, "text": text}, root=root)
    # the checkpoint of an answered turn, empty (tools were running)
    core.begin_partial(tid, "q2", root=root)
    assert core.recover_partial(tid, root=root) is None
    assert not os.path.isfile(core.partial_path(tid, root=root))
    # ... or of an EARLIER answered turn (a mirror taken during q1)
    core.begin_partial(tid, "q1", root=root)
    assert core.recover_partial(tid, root=root) is None
    assert [e["role"] for e in core.load_transcript(tid, root)] == \
        ["user", "assistant", "user", "assistant"]


def test_a_real_interruption_is_still_recovered(tmp_path):
    root, tid = str(tmp_path), "B2REAL"
    # the same words were asked and answered before; this time no answer
    for role, text in (("user", "continue"), ("assistant", "done 1"),
                       ("user", "continue")):
        core.append_transcript(tid, {"role": role, "text": text}, root=root)
    core.begin_partial(tid, "continue", root=root)
    core.checkpoint_partial(tid, "half an answer", root=root)
    rec = core.recover_partial(tid, root=root)
    assert rec is not None and "half an answer" in rec["text"]


def test_partial_is_stale_rules():
    done = [{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}]
    assert core.partial_is_stale({"prompt": "q"}, done)
    assert core.partial_is_stale({}, done)                  # names no question
    assert not core.partial_is_stale({"prompt": "q"}, done[:1])   # unanswered
    assert not core.partial_is_stale({"prompt": "q"}, [])
    # answered last turn, but this question was never in the transcript
    assert not core.partial_is_stale({"prompt": "other"}, done)


# ---------------------------------------------------------------------------
# B10: the file note names a chart's picture
# ---------------------------------------------------------------------------

def test_the_file_note_names_a_charts_png_beside_its_sidecar(tmp_path):
    files = tmp_path / "conv" / "files"
    files.mkdir(parents=True)
    side = files / "wc_vs_depth.plotly.json"
    side.write_text("{}")
    (files / "wc_vs_depth.png").write_bytes(b"\x89PNG" + b"p" * 50)
    lone = files / "pressure.plotly.json"              # no picture written
    lone.write_text("{}")
    tr = [{"role": "user", "text": "plot it"},
          {"role": "assistant", "text": "done",
           "artifacts": [str(side), str(lone)]}]
    note = core.working_files_note(str(files), tr)
    lines = note.splitlines()
    png = [ln for ln in lines if ln.startswith("- 'wc_vs_depth.png'")]
    js = [ln for ln in lines if ln.startswith("- 'wc_vs_depth.plotly.json'")]
    assert png and js, note
    assert "you produced it in turn 1" in png[0]
    assert "'wc_vs_depth.plotly.json'" in png[0] and "documents" in png[0]
    assert "its picture is 'wc_vs_depth.png'" in js[0]
    assert lines.index(png[0]) < lines.index(js[0])
    lone_row = [ln for ln in lines if "'pressure.plotly.json'" in ln]
    assert lone_row and "picture" not in lone_row[0]
    assert core.plotly_picture_twin(str(lone)) is None
    assert core.plotly_picture_twin(str(files / "wc_vs_depth.png")) is None


# ---------------------------------------------------------------------------
# B11: the files stay in the title
# ---------------------------------------------------------------------------

def test_files_and_question_title():
    t = core.files_and_question_title
    assert t("3000.pdf", "What is this sheet?") == \
        f"3000.pdf {DASH} What is this sheet?"
    assert t("3000.pdf", "Which sheets carry the GCE tag and where") == \
        f"3000.pdf {DASH} Which sheets carry the GCE tag…"
    # the question already names the file: no repeat
    assert t("3000.pdf", "Summarise 3000.pdf for me") == \
        "Summarise 3000.pdf for me"
    assert t("", "What is this?") == "What is this?"
    assert t("New conversation", "What is this?") == "What is this?"
    assert t("a.pdf + b.pdf", "") == "a.pdf + b.pdf"


def test_two_conversations_on_two_sheets_keep_distinct_titles_and_folders(
        monkeypatch):
    """F12: both conversations were "What is this sheet?" in the sidebar."""
    fm = RemoteFM()
    store = sp.SharePointStore(file_manager=fm)
    titles, folders = [], []
    for tid, sheet in (("B11SHEET3000", "3000.pdf"),
                       ("B11SHEET3001", "3001.pdf")):
        core.ensure_conversation(tid)
        files = core.conversation_files_dir(tid)
        os.makedirs(files, exist_ok=True)
        transcript = []
        _run_turn(monkeypatch, tid, files, transcript,
                  f"I just attached `{sheet}`. Give me a short orientation.",
                  orientation=[sheet])
        store.mirror_conversation(tid)
        pinned = core.load_meta(tid)[sp.MIRROR_FOLDER_KEY]
        assert pinned.startswith(sheet + "_")
        _run_turn(monkeypatch, tid, files, transcript, "What is this sheet?")
        store.mirror_conversation(tid)
        meta = core.load_meta(tid)
        assert meta[sp.MIRROR_FOLDER_KEY] == pinned          # never moves
        assert store.session_folder(tid).endswith("/" + pinned)
        titles.append(meta["title"])
        folders.append(pinned)
    assert titles == [f"3000.pdf {DASH} What is this sheet?",
                      f"3001.pdf {DASH} What is this sheet?"]
    assert len(set(folders)) == 2


# ---------------------------------------------------------------------------
# B12: a new folder name is unique against the remote library
# ---------------------------------------------------------------------------

def _titled(tid, title, created, root=None):
    core.ensure_conversation(tid, title=title, root=root)
    meta = core.load_meta(tid, root)
    meta["created"] = created
    core.save_meta(tid, meta, root)
    return meta


def test_a_folder_another_host_made_gets_a_thread_shard(tmp_path):
    """F05/F06 chose the same ``<file>_<date>`` folder; on a wiped host or a
    second host the local dedupe cannot see the first one."""
    fm = RemoteFM()
    store = sp.SharePointStore(file_manager=fm)
    created = time.time()
    meta = _titled("NEWTHREAD999", "harbour_road.pdf", created)
    want = f"harbour_road.pdf_{sp.folder_date(meta)}"
    base = store.conversations_base()
    fm.put(f"{base}/{want}/meta.json", {"thread_id": "OTHERHOST111"})
    fm.put(f"{base}/{want}/transcript.jsonl", b"theirs")
    out = store.mirror_conversation("NEWTHREAD999")
    assert not out["errors"]
    assert out["folder"] == f"{base}/{want}_NEWTHR"
    assert core.load_meta("NEWTHREAD999")[sp.MIRROR_FOLDER_KEY] == \
        f"{want}_NEWTHR"
    assert not any(u.startswith(f"{base}/{want}/") for u in fm.uploads)
    assert fm.tree[f"{base}/{want}/transcript.jsonl"] == b"theirs"


def test_the_remote_check_is_case_insensitive_and_keeps_its_own_folder():
    fm = RemoteFM()
    store = sp.SharePointStore(file_manager=fm)
    created = time.time()
    meta = _titled("CASEDIFF0001", "Plans.PDF", created)
    base = store.conversations_base()
    fm.put(f"{base}/plans.pdf_{sp.folder_date(meta)}/meta.json",
           {"thread_id": "SOMEONEELSE1"})
    assert store.mirror_conversation("CASEDIFF0001")["folder"].endswith(
        "_CASEDI")
    # a folder that is this conversation's own keeps its name
    meta2 = _titled("MINEMINE0001", "Report.pdf", created)
    own = f"Report.pdf_{sp.folder_date(meta2)}"
    fm.put(f"{base}/{own}/meta.json", {"thread_id": "MINEMINE0001"})
    assert store.mirror_conversation("MINEMINE0001")["folder"] == \
        f"{base}/{own}"


def test_an_unreadable_library_gets_the_shard_and_a_fixed_folder_never_moves():
    class Broken(RemoteFM):
        def ls(self, path):
            raise RuntimeError("no token")
    fm = Broken()
    store = sp.SharePointStore(file_manager=fm)
    _titled("BROKENLS0001", "site.pdf", time.time())
    assert store.mirror_conversation("BROKENLS0001")["folder"].endswith(
        "_BROKEN")

    # an existing conversation whose folder is fixed is never re-checked
    fm2 = RemoteFM()
    store2 = sp.SharePointStore(file_manager=fm2)
    meta = _titled("PINNED000001", "site.pdf", time.time())
    meta[sp.MIRROR_FOLDER_KEY] = "site.pdf_2026-10-01"
    core.save_meta("PINNED000001", meta)
    base = store2.conversations_base()
    fm2.put(f"{base}/site.pdf_2026-10-01/meta.json", {"thread_id": "ELSE"})
    out = store2.mirror_conversation("PINNED000001")
    assert out["folder"] == f"{base}/site.pdf_2026-10-01"
    assert fm2.ls_calls == []
    assert sp.conversation_folder("PINNED000001",
                                  core.load_meta("PINNED000001")) == \
        "site.pdf_2026-10-01"
