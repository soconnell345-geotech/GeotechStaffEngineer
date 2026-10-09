"""One process, many people: each conversation's files and documents are its
own (live smoke wave 1, 2026-10-08: A2, A4, A6, A7, A9).

* A2 -- ``list_files('.')`` listed the server process's folder, and a calc
  sub-agent walked other conversations: the real-disk read tools now read
  only the conversation's working folder (and the reference library) while
  a host has bound one; a relative path is inside that folder.
* A4 -- one process-wide document toolkit: handles, the open-document limit
  and the "open handles" hint were shared by everyone.
* A6 -- results named files by their server paths.
* A7 -- 29 of 43 tool errors: a file name where a handle was expected, a
  handle where an attachment key was expected.
* A9 -- page thumbnails went to a shared, never-cleaned ``%TEMP%`` folder.

Two conversations are two working folders bound in turn, the way the web
app's turn worker binds one per turn. No model, no network.
"""

import json
import os

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from funhouse_agent import _fileio, document_tools, vision_tools  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    SCRATCH_DIR, _resolve_attachment_or_path, dispatch_extended_tool)
from planlens.testing import build_synthetic_review_document  # noqa: E402


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _pdf(text: str) -> bytes:
    d = fitz.open()
    d.new_page().insert_text((72, 72), text)
    data = d.tobytes()
    d.close()
    return data


def _doc(name, args, attachments=None):
    return json.loads(document_tools.dispatch_document_tool(
        name, args, attachments=attachments or {}, max_chars=7500))


def _ext(name, args, attachments=None, save_fn=None):
    return json.loads(dispatch_extended_tool(
        name, args, engine=None, attachments=attachments or {},
        save_fn=save_fn))


@pytest.fixture
def people(tmp_path, monkeypatch):
    """Alice's and Bob's working folders, laid out like the app's; the
    server process runs in a folder holding a secret."""
    monkeypatch.delenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, raising=False)
    monkeypatch.delenv(vision_tools.EXTRA_READ_ROOTS_ENV, raising=False)
    server = tmp_path / "server"
    server.mkdir()
    (server / ".env").write_text("SECRET=1")
    monkeypatch.chdir(server)
    out = {}
    for who in ("alice", "bob"):
        files = (tmp_path / "data" / "users" / f"corp__{who}"
                 / "document_review" / "conversations" / f"{who}01" / "files")
        files.mkdir(parents=True)
        out[who] = files
    (out["alice"] / "alice_notes.txt").write_text("Alice's private notes")
    (out["alice"] / "alice_report.pdf").write_bytes(_pdf("ALICE ONLY"))
    (out["bob"] / "plan.pdf").write_bytes(_pdf("BOB PLAN"))
    (out["bob"] / "bob_notes.txt").write_text("Bob's notes")
    return out


# -- A2: the read tools read this conversation's files ------------------------

def test_dot_is_the_working_folder_not_the_server_folder(people):
    with _fileio.working_dir_bound(people["bob"]):
        out = _ext("list_files", {"path": "."})
    names = {e["name"] for e in out["entries"]}
    assert names == {"plan.pdf", "bob_notes.txt"}
    assert out["path"] == "."                       # never the server path
    raw = json.dumps(out)
    assert ".env" not in raw and str(people["bob"]) not in raw


def test_alices_files_are_invisible_to_bob(people):
    alice = people["alice"]
    with _fileio.working_dir_bound(people["bob"]):
        listed = _ext("list_files", {"path": str(alice)})
        read = _ext("read_text_file", {"path": str(alice / "alice_notes.txt")})
        climb = _ext("read_text_file",
                     {"path": "../../alice01/files/alice_notes.txt"})
        pdf = _ext("read_pdf_text", {"source": str(alice / "alice_report.pdf")})
        server = _ext("read_text_file", {"path": str(os.getcwd()) + "/.env"})
        opened = _doc("open_document",
                      {"source": str(alice / "alice_report.pdf")})
    for out in (listed, read, climb, pdf, server, opened):
        assert "error" in out, out
        text = json.dumps(out)
        assert "Alice's private" not in text and "ALICE ONLY" not in text
        assert "SECRET" not in text
        # the refusal points at Bob's own files, by name, never by path
        assert "plan.pdf" in text and str(people["bob"]) not in text


def test_relative_names_resolve_in_the_working_folder(people):
    with _fileio.working_dir_bound(people["bob"]):
        out = _ext("read_text_file", {"path": "bob_notes.txt"})
        listed = _ext("list_files", {"path": "plan.pdf"})
    assert out["text"] == "Bob's notes" and out["path"] == "bob_notes.txt"
    assert listed["is_file"] and listed["path"] == "plan.pdf"


def test_a_library_caller_with_no_folder_keeps_any_path(people):
    out = _ext("read_text_file",
               {"path": str(people["alice"] / "alice_notes.txt")})
    assert out["text"] == "Alice's private notes"
    assert _ext("list_files", {"path": "."})["n_entries"] == 1   # the cwd


def test_reference_pdfs_are_read_by_name(people, tmp_path, monkeypatch):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "ufc_3_220_20.pdf").write_bytes(_pdf("UFC CHART"))
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(refs))
    with _fileio.working_dir_bound(people["bob"]):
        data, kind = _resolve_attachment_or_path("ufc_3_220_20.pdf", {})
        assert kind == "path" and data[:4] == b"%PDF"
        assert vision_tools.display_path(str(refs / "ufc_3_220_20.pdf")) \
            == "ufc_3_220_20.pdf"


def test_a_save_outside_lands_in_the_working_folder_named_by_name(people):
    alice = people["alice"]
    with _fileio.working_dir_bound(people["bob"]):
        out = _ext("save_file", {"path": str(alice / "planted.txt"),
                                 "content": "x"})
    assert out["saved"] == "planted.txt"
    assert (people["bob"] / "planted.txt").is_file()
    assert not (alice / "planted.txt").exists()


def test_a_hosts_saved_note_survives_the_short_name(people):
    seen = []

    def host_save(path, content):
        p = path if os.path.isabs(path) else os.path.join(
            str(people["bob"]), os.path.basename(path))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        return p

    host_save.saved_note = lambda p: seen.append(p) or "Attached to the chat."
    with _fileio.working_dir_bound(people["bob"]):
        out = _ext("save_file", {"path": "memo.md", "content": "# m"},
                   save_fn=host_save)
    assert out["saved"] == "memo.md"
    assert out["note"] == "Attached to the chat."
    assert seen == [str(people["bob"] / "memo.md")]    # asked about the REAL path


def test_a_marked_up_copy_never_goes_into_another_folder(people):
    with _fileio.working_dir_bound(people["bob"]):
        where = document_tools.markup_output_path(
            str(people["alice"] / "x_marked.pdf"))
    assert where == str(people["bob"] / "x_marked.pdf")


# -- A4: a document handle belongs to the conversation that opened it --------

def test_alices_handle_is_invisible_to_bob(people, gt):
    with _fileio.working_dir_bound(people["alice"]):
        h_alice = _doc("open_document", {"source": "set.pdf"},
                       {"set.pdf": gt.pdf})["handle"]
    with _fileio.working_dir_bound(people["bob"]):
        out = _doc("read_document", {"handle": h_alice})
        h_bob = _doc("open_document", {"source": "plan.pdf"})["handle"]
        miss = _doc("read_document", {"handle": "doc_0000000000"})
    assert "error" in out and "Alice" not in json.dumps(out)
    assert h_alice not in out.get("hint", "")      # echoed, never listed
    # the "open handles" hint names Bob's documents only
    assert h_bob in miss["hint"] and h_alice not in miss["hint"]
    with _fileio.working_dir_bound(people["alice"]):
        assert "error" not in _doc("read_document", {"handle": h_alice})


def test_the_open_document_limit_is_per_conversation(people):
    with _fileio.working_dir_bound(people["bob"]):
        h_bob = _doc("open_document", {"source": "plan.pdf"})["handle"]
    with _fileio.working_dir_bound(people["alice"]):
        for i in range(10):                  # more than planlens keeps open
            _doc("open_document", {"source": f"a{i}.pdf"},
                 {f"a{i}.pdf": _pdf(f"alice document {i}")})
    with _fileio.working_dir_bound(people["bob"]):
        assert h_bob in document_tools._toolkit()._entries
        assert "BOB PLAN" in json.dumps(_doc("read_document",
                                             {"handle": h_bob}))


def test_idle_toolkits_close_and_their_handles_reopen(people, monkeypatch):
    monkeypatch.setattr(document_tools, "MAX_LIVE_TOOLKITS", 1)
    with _fileio.working_dir_bound(people["alice"]):
        h = _doc("open_document", {"source": "alice_report.pdf"})["handle"]
        alice_space = document_tools._space()
    with _fileio.working_dir_bound(people["bob"]):
        _doc("open_document", {"source": "plan.pdf"})
    assert alice_space.kit is None           # closed: memory stays bounded
    with _fileio.working_dir_bound(people["alice"]):
        out = _doc("read_document", {"handle": h})
    assert "ALICE ONLY" in json.dumps(out) and "opened" not in out


# -- A7: a handle and a source are interchangeable ---------------------------

def test_a_file_name_works_where_a_handle_goes(people, gt):
    atts = {"submittal.pdf": gt.pdf}
    with _fileio.working_dir_bound(people["bob"]):
        # read before any open_document: the parallel-batch case
        out = _doc("search_document", {"handle": "submittal.pdf",
                                       "pattern": "embedment"}, atts)
        h = _doc("open_document", {"source": "submittal.pdf"}, atts)["handle"]
        bad = _doc("read_document", {"handle": "nothing_like_this.pdf"}, atts)
    assert out["n_hits"] == 2 and h in out["opened"]
    assert "did not open" in bad["error"] and "submittal.pdf" in bad["hint"]


def test_a_handle_works_where_a_file_goes(people, gt):
    atts = {"submittal.pdf": gt.pdf}
    with _fileio.working_dir_bound(people["bob"]):
        h = _doc("open_document", {"source": "submittal.pdf"}, atts)["handle"]
        again = _doc("open_document", {"source": h}, atts)
        text = _ext("read_pdf_text", {"source": h}, atts)
        data, kind = _resolve_attachment_or_path(h, atts)
        file_h = _doc("open_document", {"source": "plan.pdf"})["handle"]
        file_data, file_kind = _resolve_attachment_or_path(file_h, {})
    assert again["handle"] == h and again["name"] == "submittal.pdf"
    assert text["n_pages_total"] == 5 and "error" not in text
    assert kind == "attachment" and data == gt.pdf
    assert file_kind == "path" and file_data[:4] == b"%PDF"


def test_another_conversations_handle_is_not_a_file_here(people, gt):
    with _fileio.working_dir_bound(people["alice"]):
        h = _doc("open_document", {"source": "set.pdf"},
                 {"set.pdf": gt.pdf})["handle"]
    with _fileio.working_dir_bound(people["bob"]):
        with pytest.raises(FileNotFoundError):
            _resolve_attachment_or_path(h, {})


# -- A9 + A6: scratch images, one place per conversation, short names --------

def test_thumbnails_go_to_the_conversations_scratch(people, gt):
    with _fileio.working_dir_bound(people["bob"]):
        h = _doc("open_document", {"source": "s.pdf"},
                 {"s.pdf": gt.pdf})["handle"]
        thumbs = _doc("render_page_thumbnails", {"handle": h, "pages": "0-3"})
        name = thumbs["sheets"][0]["image_path"]
        data, kind = _resolve_attachment_or_path(name, {})
        listed = _ext("list_files", {"path": "."})
    assert name.startswith(f"{SCRATCH_DIR}/") and "planlens_" not in name
    assert (people["bob"] / name).is_file()
    assert kind == "path" and data[:8] == b"\x89PNG\r\n\x1a\n"
    assert SCRATCH_DIR not in {e["name"] for e in listed["entries"]}
    # Alice's conversation has its own scratch, untouched
    assert not (people["alice"] / SCRATCH_DIR).exists()


# -- Databricks: the owner's own data folders stay readable -------------------

def test_databricks_reads_volumes_and_workspace_tiny_apps_does_not(
        people, tmp_path, monkeypatch):
    """Single-user Databricks (the owner's Funhouse cluster) reads Unity
    Catalog volumes and the workspace; Tiny Apps -- many people in one
    process, not Databricks -- stays confined to the conversation."""
    assert vision_tools.DATABRICKS_READ_ROOTS == ("/Volumes", "/Workspace")
    volumes = tmp_path / "Volumes"
    (volumes / "proj").mkdir(parents=True)
    (volumes / "proj" / "borings.txt").write_text("B-1 at 3 m")
    monkeypatch.setattr(vision_tools, "DATABRICKS_READ_ROOTS",
                        (str(volumes), str(tmp_path / "Workspace")))
    target = str(volumes / "proj" / "borings.txt")

    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    with _fileio.working_dir_bound(people["bob"]):
        refused = _ext("read_text_file", {"path": target})
    assert "outside this conversation" in refused["error"]

    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    with _fileio.working_dir_bound(people["bob"]):
        out = _ext("read_text_file", {"path": target})
        listed = _ext("list_files", {"path": str(volumes / "proj")})
        alice = _ext("read_text_file",
                     {"path": str(people["alice"] / "alice_notes.txt")})
    assert out["text"] == "B-1 at 3 m" and out["path"] == target
    assert {e["name"] for e in listed["entries"]} == {"borings.txt"}
    assert "error" in alice           # the conversation rule still holds


def test_on_databricks_the_real_roots_are_the_data_folders(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
    monkeypatch.delenv(vision_tools.EXTRA_READ_ROOTS_ENV, raising=False)
    roots = vision_tools._extra_read_roots()
    assert roots == [os.path.abspath("/Volumes"), os.path.abspath("/Workspace")]
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION")
    assert vision_tools._extra_read_roots() == []
