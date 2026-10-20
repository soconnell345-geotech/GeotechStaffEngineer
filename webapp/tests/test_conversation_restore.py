"""Restore a conversation from its downloaded zip (BACKLOG INF-45).

On Foundry a container rebuild loses the conversations folder; the zip from
Download this conversation is the user's copy. The sidebar takes it back
into this person's conversations under its own thread id, refusing any zip
that is not one of the export's or that could write outside the folder, and
a record-alone zip says in the conversation what it left out.
"""

import io
import json
import os
import shutil
import stat
import zipfile

import pytest

pytest.importorskip("streamlit")

import webapp.core as core
from webapp import conversation_export, conversation_restore, profiles
from webapp.tests.test_conversation_export import WHEN, _conversation, _write


def _folder(root, tid):
    return core.conversation_dir(tid, root)


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _restored_files(root, tid):
    """``{path in the folder: bytes}`` of what an export would take."""
    included, _left = conversation_export.collect(_folder(root, tid))
    return {rel: _read(path) for rel, path in included}


def _no_trace_in(root):
    """No conversation and no staging folder left under ``root``."""
    assert core.list_conversations(root) == []
    if os.path.isdir(root):
        assert not [n for n in os.listdir(root) if n.startswith(".restoring")]


def _zip(entries, meta=None, top="a_footing_2026-10-20"):
    """A zip of ``entries`` ({path in the folder: text}) under ``top/``,
    with a meta.json and activity.jsonl unless ``entries`` says otherwise."""
    files = {"meta.json": json.dumps(meta or {"thread_id": "abc123",
                                              "title": "A footing"}),
             "activity.jsonl": '{"event": "x"}\n'}
    files.update(entries)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for rel, text in files.items():
            if text is not None:
                zf.writestr(f"{top}/{rel}", text)
    return buf.getvalue()


def _with_entry(info, data=b"x", base=None):
    """``base`` (a conversation zip) plus one raw entry ``info``."""
    buf = io.BytesIO(base or _zip({}))
    with zipfile.ZipFile(buf, "a") as zf:
        zf.writestr(info, data)
    return buf.getvalue()


# -- the round trip ---------------------------------------------------------


def test_export_then_restore_gives_the_same_files(monkeypatch, tmp_path):
    tid, conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, when=WHEN)
    other = str(tmp_path / "rebuilt")
    res = conversation_restore.restore(x.data, root=other, zip_name=x.name)
    assert res.status == "restored", res.message
    assert res.thread_id == tid and not res.partial
    assert sorted(res.files) == sorted(x.files)
    assert _restored_files(other, tid) == _restored_files(None, tid)
    # the same conversation: its id and meta, listed like any other
    assert [m["thread_id"] for m in core.list_conversations(other)] == [tid]
    assert core.load_meta(tid, other) == core.load_meta(tid)
    # files keep the times the zip records (it keeps them to 2 s)
    for rel in ("files/calc_package.pdf", "transcript.jsonl"):
        assert abs(os.path.getmtime(os.path.join(conv, rel))
                   - os.path.getmtime(os.path.join(_folder(other, tid), rel))
                   ) <= 2.0
    assert "restored" in res.message.lower() and "Strip footing" in \
        res.message


def test_after_a_rebuild_it_comes_back_where_it_was(monkeypatch, tmp_path):
    tid, conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, when=WHEN)
    before = _restored_files(str(tmp_path), tid)
    transcript = core.load_transcript(tid)
    shutil.rmtree(conv)                              # the container rebuild
    assert core.list_conversations() == []
    res = conversation_restore.restore(x.data, root=str(tmp_path))
    assert res.status == "restored", res.message
    assert _restored_files(str(tmp_path), tid) == before
    assert core.load_transcript(tid) == transcript
    assert [m["thread_id"] for m in core.list_conversations()] == [tid]


def test_one_already_here_is_replaced_only_when_asked(monkeypatch, tmp_path):
    tid, conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, when=WHEN)
    _write(os.path.join(conv, "files", "later.csv"), "made after the zip")
    res = conversation_restore.restore(x.data, root=str(tmp_path))
    assert res.status == "exists" and "Replace the existing copy" in \
        res.message
    assert os.path.isfile(os.path.join(conv, "files", "later.csv"))
    res = conversation_restore.restore(x.data, root=str(tmp_path),
                                       replace=True)
    assert res.status == "restored" and res.replaced, res.message
    assert not os.path.exists(os.path.join(conv, "files", "later.csv"))
    # the copy that was here went to the trash, as a delete does
    trash = os.path.join(str(tmp_path), ".trash")
    (kept,) = os.listdir(trash)
    assert kept.startswith(tid) and os.path.isfile(
        os.path.join(trash, kept, "files", "later.csv"))


def test_a_running_turn_is_never_replaced(monkeypatch, tmp_path):
    tid, conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, when=WHEN)
    res = conversation_restore.restore(x.data, root=str(tmp_path),
                                       replace=True, busy=lambda t: t == tid)
    assert res.status == "refused" and "turn is running" in res.message
    assert os.path.isfile(os.path.join(conv, "partial.json"))


# -- refusals ---------------------------------------------------------------


def _link():
    info = zipfile.ZipInfo("a_footing_2026-10-20/files/passwd")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return _with_entry(info, b"/etc/passwd")


def _raw_name(name):
    info = zipfile.ZipInfo("placeholder")
    info.filename = name                  # as written, on every platform
    return _with_entry(info)


def _encrypted():
    # zipfile clears the flag when it writes, so set it in the central
    # directory record (flag at +8, name at +46) afterwards
    name = b"a_footing_2026-10-20/files/x.txt"
    data = bytearray(_with_entry(zipfile.ZipInfo(name.decode())))
    i = data.find(b"PK\x01\x02")
    while data[i + 46:i + 46 + len(name)] != name:
        i = data.find(b"PK\x01\x02", i + 1)
    data[i + 8] |= 0x1
    return bytes(data)


REFUSALS = {
    "not a zip": (lambda: b"PK? not really a zip", "not a zip"),
    "two folders": (lambda: _with_entry(zipfile.ZipInfo("other/x.txt")),
                    "one folder"),
    "loose files": (lambda: _with_entry(zipfile.ZipInfo("x.txt")),
                    "one folder"),
    "no activity": (lambda: _zip({"activity.jsonl": None}),
                    "no activity.jsonl"),
    "no meta": (lambda: _zip({"meta.json": None}), "no meta.json"),
    "zip slip": (lambda: _raw_name("a_footing_2026-10-20/../evil.txt"),
                 "leads out of its folder"),
    "absolute": (lambda: _raw_name("/a_footing_2026-10-20/evil.txt"),
                 "absolute path"),
    "drive": (lambda: _raw_name("C:/a_footing_2026-10-20/evil.txt"),
              "absolute path"),
    "backslash": (lambda: _raw_name("a_footing_2026-10-20\\..\\evil.txt"),
                  "backslash"),
    "link": (_link, "a link"),
    "encrypted": (_encrypted, "encrypted"),
    "no thread id": (lambda: _zip({}, meta={"title": "x"}),
                     "no usable conversation id"),
    "thread id is a path": (lambda: _zip({}, meta={"thread_id": "../x"}),
                            "no usable conversation id"),
    "meta not json": (lambda: _zip({"meta.json": "{oops"}),
                      "meta.json could not be read"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_each_refusal_says_why_and_writes_nothing(monkeypatch, tmp_path,
                                                  case):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    make, says = REFUSALS[case]
    root = str(tmp_path / "user")
    res = conversation_restore.restore(make(), root=root)
    assert res.status == "refused", (case, res)
    assert says in res.message, (case, res.message)
    _no_trace_in(root)
    assert not os.path.exists(tmp_path / "evil.txt")
    assert not os.path.exists(tmp_path / "user" / "evil.txt")


def test_over_the_size_cap_it_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv(conversation_restore.MAX_MB_ENV, "0.01")   # ~10 KB
    root = str(tmp_path)
    res = conversation_restore.restore(
        _zip({"files/big.bin": "0" * 20_000}), root=root)
    assert res.status == "refused" and "MB a restore takes" in res.message
    assert conversation_restore.MAX_MB_ENV in res.message
    _no_trace_in(root)
    monkeypatch.delenv(conversation_restore.MAX_MB_ENV)
    assert conversation_restore.max_bytes() == 200 * 1048576


def test_the_size_is_counted_as_it_is_written(monkeypatch, tmp_path):
    # a header that understates the size does not get past the cap
    monkeypatch.setenv(conversation_restore.MAX_MB_ENV, "0.01")
    real = conversation_restore._read_layout

    def understated(zf, cap):
        return real(zf, cap * 100)

    monkeypatch.setattr(conversation_restore, "_read_layout", understated)
    res = conversation_restore.restore(
        _zip({"files/big.bin": "0" * 20_000}), root=str(tmp_path))
    assert res.status == "refused" and "more than" in res.message
    _no_trace_in(str(tmp_path))


def test_another_pages_conversation_is_refused(tmp_path):
    data = _zip({}, meta={"thread_id": "abc123", "page": "document_review"})
    res = conversation_restore.restore(data, root=str(tmp_path),
                                       page="geotech")
    assert res.status == "refused" and "'document_review' page" in \
        res.message
    ok = conversation_restore.restore(
        _zip({}, meta={"thread_id": "abc123", "page": "geotech"}),
        root=str(tmp_path), page="geotech")
    assert ok.status == "restored", ok.message


def test_one_open_under_another_root_is_refused(tmp_path):
    core.register_thread_root("abc123", str(tmp_path / "someone_else"))
    try:
        res = conversation_restore.restore(_zip({}), root=str(tmp_path))
    finally:
        core.register_thread_root("abc123", None)
    assert res.status == "refused" and "another page or account" in \
        res.message
    _no_trace_in(str(tmp_path))


def test_what_the_export_leaves_out_is_not_restored(tmp_path):
    data = _zip({"sp_manifest.json": "{}", "partial.json": "{}",
                 "files/.env": "KEY=1", "files/client.pem": "-----BEGIN",
                 "files/big.csv.part": "half", "files/ok.csv": "a,b"})
    res = conversation_restore.restore(data, root=str(tmp_path))
    assert res.status == "restored", res.message
    folder = _folder(str(tmp_path), "abc123")
    assert sorted(os.listdir(folder)) == ["activity.jsonl", "files",
                                          "meta.json"]
    assert os.listdir(os.path.join(folder, "files")) == ["ok.csv"]
    assert {rel for rel, _w in res.skipped} == {
        "sp_manifest.json", "partial.json", "files/.env",
        "files/client.pem", "files/big.csv.part"}


def test_the_restored_conversation_is_this_persons(tmp_path):
    meta = {"thread_id": "abc123", "owner": "dom__alice",
            "owner_name": "Alice", "page": "geotech"}
    res = conversation_restore.restore(_zip({}, meta=meta),
                                       root=str(tmp_path / "bob"),
                                       owner="dom__bob", owner_name="Bob")
    assert res.status == "restored", res.message
    got = core.load_meta("abc123", str(tmp_path / "bob"))
    assert (got["owner"], got["owner_name"]) == ("dom__bob", "Bob")
    # a single-user host tags no owner
    res = conversation_restore.restore(_zip({}, meta=meta),
                                       root=str(tmp_path / "solo"))
    assert "owner" not in core.load_meta("abc123", str(tmp_path / "solo"))


# -- the record-alone zip ---------------------------------------------------


def _partial(monkeypatch, tmp_path):
    tid, conv = _conversation(monkeypatch, tmp_path)
    full = conversation_export.build(tid, when=WHEN)
    _write(os.path.join(conv, "files", "model_run.bin"), os.urandom(60_000))
    x = conversation_export.build(tid, cap=full.size + 40_000,
                                  record_only=True, when=WHEN)
    assert x.data is not None and "files/model_run.bin" in dict(x.left_out)
    return tid, x


def test_a_record_alone_zip_restores_what_it_has_and_says_so(monkeypatch,
                                                             tmp_path):
    tid, x = _partial(monkeypatch, tmp_path)
    other = str(tmp_path / "rebuilt")
    res = conversation_restore.restore(x.data, root=other, zip_name=x.name,
                                       when=WHEN)
    assert res.status == "restored" and res.partial, res.message
    assert "partial copy" in res.message
    assert sorted(res.files) == sorted(x.files)
    folder = _folder(other, tid)
    assert not os.path.exists(os.path.join(folder, "LEFT_OUT.txt"))
    assert not os.path.exists(os.path.join(folder, "files", "model_run.bin"))
    # the conversation says what was left out, as its last entry
    note = core.load_transcript(tid, other)[-1]
    assert note["role"] == "assistant" and note["restored"]
    assert x.name in note["text"] and "2026-10-20" in note["text"]
    assert "- `files/model_run.bin`: did not fit" in note["text"]
    assert "- `files/report.pdf`: an upload" in note["text"]
    # and the agent hears it on the next turn
    assert "files/model_run.bin" in res.agent_note()
    assert set(res.left_out) >= {"files/model_run.bin", "files/report.pdf"}


def test_a_full_zip_gives_the_agent_no_note(monkeypatch, tmp_path):
    tid, _conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, when=WHEN)
    res = conversation_restore.restore(x.data, root=str(tmp_path / "b"))
    assert res.status == "restored" and res.agent_note() is None


def test_a_record_alone_zip_may_lack_the_activity_log(tmp_path):
    note = ("This zip holds ...\n\nLeft out:\n"
            "  activity.jsonl: did not fit (30.0 MB)\n")
    res = conversation_restore.restore(
        _zip({"activity.jsonl": None, "LEFT_OUT.txt": note}),
        root=str(tmp_path))
    assert res.status == "restored" and res.partial, res.message
    assert res.left_out == ["activity.jsonl"]
    # a zip that only claims to be partial still needs its activity log
    res = conversation_restore.restore(
        _zip({"activity.jsonl": None, "LEFT_OUT.txt": "Left out:\n"}),
        root=str(tmp_path / "b"))
    assert res.status == "refused"


# -- the sidebar ------------------------------------------------------------


class _Upload(io.BytesIO):
    """What st.file_uploader hands back, enough for the box."""

    def __init__(self, data, name, file_id):
        super().__init__(data)
        self.name, self.size, self.file_id = name, len(data), file_id


def test_the_box_is_the_geotech_pages():
    assert profiles.GEOTECH.conversation_download
    assert not profiles.DOCUMENT_REVIEW.conversation_download


def test_the_sidebar_restores_the_zip_and_opens_it(monkeypatch, tmp_path):
    import streamlit
    from webapp.tests.test_app_smoke import _mk_at, _stream_ok
    tid, x = _partial(monkeypatch, tmp_path / "before")
    calls = []
    real = streamlit.file_uploader

    def uploader(label, *a, key=None, **k):
        if str(key) == "restore_zip_0":
            calls.append(key)
            return _Upload(x.data, x.name, "f1")
        return real(label, *a, key=key, **k)

    monkeypatch.setattr(streamlit, "file_uploader", uploader)
    at = _mk_at(monkeypatch, tmp_path / "after", _stream_ok).run()
    assert not at.exception, [e.value for e in at.exception]
    # restored once, opened, and the box cleared for the next zip
    assert at.session_state["thread_id"] == tid
    assert at.session_state["restore_epoch"] == 1
    assert core.load_meta(tid, str(tmp_path / "after")) is not None
    assert at.session_state["transcript"][-1].get("restored")
    assert any("partial copy" in s.value for s in at.success)
    assert "files/model_run.bin" in at.session_state["pending_notes"][0]
    at.run()
    assert not at.exception and calls == ["restore_zip_0"]


def test_the_sidebar_replaces_only_once_ticked(monkeypatch, tmp_path):
    import streamlit
    from webapp.tests.test_app_smoke import _mk_at, _stream_ok
    tid, _conv = _conversation(monkeypatch, tmp_path / "after")
    x = conversation_export.build(tid, when=WHEN)
    real = streamlit.file_uploader

    def uploader(label, *a, key=None, **k):
        if str(key).startswith("restore_zip_"):
            return (_Upload(x.data, x.name, "f1")
                    if key == "restore_zip_0" else None)
        return real(label, *a, key=key, **k)

    monkeypatch.setattr(streamlit, "file_uploader", uploader)
    at = _mk_at(monkeypatch, tmp_path / "after", _stream_ok).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["thread_id"] != tid
    assert any("already here" in i.value for i in at.info)
    at.run()                                 # the same zip is not redone
    assert at.session_state["restore_result"]["status"] == "exists"
    (tick,) = [c for c in at.checkbox if c.label == "Replace the existing copy"]
    tick.check().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["thread_id"] == tid
    assert any("replacing the copy" in s.value for s in at.success)


def test_on_the_websocket_uploader_the_zip_is_taken_out_at_once(monkeypatch,
                                                                tmp_path):
    # a file left in that uploader is sent again on every rerun: the box
    # retires it on arrival, holds the bytes, and restores from them
    from webapp import ws_upload
    from webapp.tests.test_app_smoke import _mk_at, _stream_ok
    tid, _conv = _conversation(monkeypatch, tmp_path / "before")
    x = conversation_export.build(tid, when=WHEN)
    sent = {"restore_zip_0": [("not_a.zip", b"junk")]}
    keys = []

    def uploader(_types, key="ws_upload"):
        keys.append(key)
        return sent.pop(key, []), []

    monkeypatch.setattr(ws_upload, "upload_mode", lambda: "ws")
    monkeypatch.setattr(ws_upload, "ws_file_uploader", uploader)
    at = _mk_at(monkeypatch, tmp_path / "after", _stream_ok).run()
    assert not at.exception, [e.value for e in at.exception]
    # the junk was taken out (epoch 1), held, and refused
    assert at.session_state["restore_epoch"] == 1
    assert at.session_state["restore_held"][0] == "not_a.zip"
    assert any("not a zip" in e.value for e in at.error)
    at.run()                                 # nothing redone
    assert at.session_state["restore_epoch"] == 1
    sent["restore_zip_1"] = [(x.name, x.data)]
    at.run()                                 # the next zip arrives
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["thread_id"] == tid
    assert at.session_state["restore_epoch"] == 3
    assert "restore_held" not in at.session_state
    assert [k for k in keys if k.startswith("restore_zip_")][:2] == [
        "restore_zip_0", "restore_zip_1"]
