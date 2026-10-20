"""Download this conversation: a zip of its folder (BACKLOG INF-38).

On Foundry the conversations folder is inside the published app's container
and is lost when the container is rebuilt. The geotech page's sidebar zips
the current conversation's record, files and figures when asked, through
the normal download (the browser-made one on Foundry), leaving out caches
and anything that may hold a key; over the download limit it offers the
record alone, with the deliverables that fit.
"""

import io
import json
import os
import re
import time
import zipfile

import pytest

pytest.importorskip("streamlit")

import webapp.core as core
from webapp import conversation_export, file_widgets, profiles

WHEN = time.mktime((2026, 10, 20, 12, 0, 0, 0, 0, -1))


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def _conversation(monkeypatch, tmp_path, tid="t0123456789",
                  title="Strip footing: q_all?"):
    """A conversation folder holding every kind of file the zip sorts."""
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    meta = core.ensure_conversation(tid)
    meta["title"] = title
    core.save_meta(tid, meta)
    conv = core.conversation_dir(tid)
    record = {
        "activity.jsonl": '{"event": "tool_start"}\n' * 200,
        "transcript.jsonl": '{"role": "user", "text": "hi"}\n',
        "messages.json": "[]",
        "attachments.json": json.dumps(["report.pdf"]),
        "sp_manifest.json": "{}",                  # this machine's mirror list
        "partial.json": "{}",                      # a turn checkpoint
    }
    for name, text in record.items():
        _write(os.path.join(conv, name), text)
    files = os.path.join(conv, "files")
    _write(os.path.join(files, "calc_package.pdf"), b"%PDF-1.7 calc" * 50)
    _write(os.path.join(files, "figures", "profile.png"), b"\x89PNG" * 40)
    _write(os.path.join(files, "report.pdf"), b"%PDF-1.7 upload" * 50)
    _write(os.path.join(files, "digest", "pages.json"), "{}")     # a cache
    _write(os.path.join(files, ".scratch", "sheet.png"), b"x")    # scratch
    _write(os.path.join(files, ".env"), "PROMPTER_API_KEY=k")     # a key
    _write(os.path.join(files, "client.pem"), "-----BEGIN")       # a key
    _write(os.path.join(files, "box", "__pycache__", "a.pyc"), b"c")
    _write(os.path.join(files, "big.csv.part"), "half")           # in flight
    return tid, conv


def _names(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return set(zf.namelist())


def test_the_zip_holds_the_record_files_and_figures(monkeypatch, tmp_path):
    tid, _conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, when=WHEN)
    assert x.name == "Strip_footing_q_all_2026-10-20.zip"
    top = "Strip_footing_q_all_2026-10-20"
    assert _names(x.data) == {f"{top}/{rel}" for rel in (
        "meta.json", "activity.jsonl", "transcript.jsonl", "messages.json",
        "attachments.json", "files/calc_package.pdf",
        "files/figures/profile.png", "files/report.pdf")}
    assert sorted(x.files) == sorted(n.split("/", 1)[1]
                                     for n in _names(x.data))
    # laid out as the folder is: the meta names the conversation
    with zipfile.ZipFile(io.BytesIO(x.data)) as zf:
        assert json.loads(zf.read(f"{top}/meta.json"))["thread_id"] == tid
        assert zf.read(f"{top}/files/calc_package.pdf") == \
            b"%PDF-1.7 calc" * 50


def test_caches_keys_and_unfinished_files_are_left_out(monkeypatch,
                                                       tmp_path):
    tid, _conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, when=WHEN)
    why = dict(x.left_out)
    assert why["files/digest/"].startswith("a cache")
    assert why["files/box/__pycache__/"].startswith("a cache")
    assert why["files/.scratch/"].startswith("a cache")
    assert "key" in why["files/.env"]
    assert "key" in why["files/client.pem"]
    assert why["partial.json"] == "still being written"
    assert why["files/big.csv.part"] == "still being written"
    assert "mirror" in why["sp_manifest.json"]
    blob = x.data
    assert b"PROMPTER_API_KEY" not in blob and b"BEGIN" not in blob


def test_a_link_out_of_the_folder_is_left_out(monkeypatch, tmp_path):
    tid, conv = _conversation(monkeypatch, tmp_path)
    secret = tmp_path / "outside.txt"
    secret.write_text("not the conversation's")
    try:
        os.symlink(secret, os.path.join(conv, "files", "linked.txt"))
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot make a symlink")
    x = conversation_export.build(tid, when=WHEN)
    assert "files/linked.txt" in dict(x.left_out)
    assert not any(n.endswith("linked.txt") for n in _names(x.data))


def test_an_untitled_conversation_is_named_by_its_thread():
    assert conversation_export.zip_name(
        {"title": "New conversation"}, "abcdef0123456", WHEN) == \
        "conversation_abcdef01_2026-10-20.zip"
    assert conversation_export.zip_name(
        {"title": 'a/b "c" #1'}, "t", WHEN) == "a_b_c_1_2026-10-20.zip"


def test_over_the_limit_it_says_so_and_builds_nothing(monkeypatch, tmp_path):
    tid, _conv = _conversation(monkeypatch, tmp_path)
    x = conversation_export.build(tid, cap=200, when=WHEN)
    assert x.data is None and x.size == 0
    assert "files/calc_package.pdf" in x.files      # what it would hold
    assert x.folder_bytes > 200


def test_the_record_alone_takes_the_deliverables_that_fit(monkeypatch,
                                                          tmp_path):
    tid, conv = _conversation(monkeypatch, tmp_path)
    full = conversation_export.build(tid, when=WHEN)
    # a deliverable that cannot fit: random bytes do not compress
    _write(os.path.join(conv, "files", "model_run.bin"), os.urandom(60_000))
    cap = full.size + 40_000
    x = conversation_export.build(tid, cap=cap, record_only=True, when=WHEN)
    assert x.record_only and x.data is not None and x.size <= cap
    top = x.name[:-4]
    names = _names(x.data)
    for rel in ("meta.json", "activity.jsonl", "transcript.jsonl",
                "files/calc_package.pdf", "files/figures/profile.png",
                conversation_export.NOTE_NAME):
        assert f"{top}/{rel}" in names, rel
    why = dict(x.left_out)
    assert why["files/model_run.bin"].startswith("did not fit")
    assert why["files/report.pdf"].startswith("an upload")
    assert f"{top}/files/report.pdf" not in names
    with zipfile.ZipFile(io.BytesIO(x.data)) as zf:
        note = zf.read(f"{top}/{conversation_export.NOTE_NAME}").decode()
    assert "files/model_run.bin: did not fit" in note
    assert "files/report.pdf: an upload" in note
    # the record goes first: meta.json and activity.jsonl are the zip's head
    with zipfile.ZipFile(io.BytesIO(x.data)) as zf:
        head = [n.split("/", 1)[1] for n in zf.namelist()[:2]]
    assert head == ["meta.json", "activity.jsonl"]


def test_the_limit_is_the_foundry_downloads_and_none_elsewhere(monkeypatch):
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    assert file_widgets.download_limit_bytes() is None
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv(file_widgets.DOWNLOAD_MAX_MB_ENV, "2")
    assert file_widgets.download_limit_bytes() == 2 * 1024 * 1024


def test_the_button_is_the_geotech_pages():
    assert profiles.GEOTECH.conversation_download
    assert not profiles.DOCUMENT_REVIEW.conversation_download


# -- the sidebar ----------------------------------------------------------


def _stream_files(_agent, _messages, thread_id, **_kw):
    files = core.conversation_files_dir(thread_id)
    with open(os.path.join(files, "inputs.csv"), "w", encoding="utf-8") as fh:
        fh.write("phi,30\n")
    yield {"kind": "token", "text": "Done."}
    yield {"kind": "turn_done", "answer": "Done.", "turn_tokens": 2}


def _app_after_a_turn(monkeypatch, tmp_path, calls):
    from webapp.tests.test_app_smoke import _mk_at
    real = conversation_export.build

    def counted(*a, **k):
        calls.append(k)
        return real(*a, **k)

    monkeypatch.setattr(conversation_export, "build", counted)
    at = _mk_at(monkeypatch, tmp_path, _stream_files).run()
    at.chat_input[0].set_value("a strip footing").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _button(at, label):
    (button,) = [b for b in at.button if b.label == label]
    return button


def test_nothing_is_zipped_until_prepare_is_pressed(monkeypatch, tmp_path):
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    calls = []
    at = _app_after_a_turn(monkeypatch, tmp_path, calls)
    assert calls == [] and "conv_zip" not in at.session_state
    assert not any(b.proto.label.endswith(".zip")
                   for b in at.get("download_button"))
    _button(at, "Prepare download").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(calls) == 1 and calls[0]["cap"] is None
    (zipped,) = [b for b in at.get("download_button")
                 if b.proto.label.endswith(".zip")]
    assert re.fullmatch(r"Download a_strip_footing_\d{4}-\d\d-\d\d\.zip",
                        zipped.proto.label)
    # an ordinary rerun keeps it and builds nothing more
    at.run()
    assert len(calls) == 1
    # a new turn makes it out of date: it goes, until prepared again
    at.chat_input[0].set_value("and the settlement?").run()
    assert "conv_zip" not in at.session_state
    assert not any(b.proto.label.endswith(".zip")
                   for b in at.get("download_button"))


def test_on_foundry_the_zip_is_the_browser_download(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    pages = []
    monkeypatch.setattr(file_widgets, "_component",
                        lambda page, height: pages.append(page))
    calls = []
    at = _app_after_a_turn(monkeypatch, tmp_path, calls)
    assert calls == []
    assert not any(".zip" in p for p in pages)
    pages.clear()
    _button(at, "Prepare download").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls[0]["cap"] == file_widgets.download_cap_bytes()
    names = [json.loads(re.search(r"const NAME = (.*);", p).group(1))
             for p in pages]
    assert any(n.startswith("a_strip_footing_") and n.endswith(".zip")
               for n in names)


def test_on_foundry_over_the_limit_the_record_alone_is_offered(monkeypatch,
                                                               tmp_path):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    pages = []
    monkeypatch.setattr(file_widgets, "_component",
                        lambda page, height: pages.append(page))
    calls = []
    at = _app_after_a_turn(monkeypatch, tmp_path, calls)
    # a limit the whole conversation passes but its record does not
    monkeypatch.setenv(file_widgets.DOWNLOAD_MAX_MB_ENV, "0.03")
    files = core.conversation_files_dir(at.session_state["thread_id"])
    _write(os.path.join(files, "big_upload.bin"), os.urandom(40_000))
    pages.clear()
    _button(at, "Prepare download").click().run()
    assert not at.exception, [e.value for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "zips to more than the 0.03 MB" in captions
    assert not any(".zip" in p for p in pages)
    _button(at, "Prepare the record alone").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls[-1]["record_only"] is True
    assert any(".zip" in p for p in pages)
    captions = " ".join(c.value for c in at.caption)
    assert "LEFT_OUT.txt" in captions
