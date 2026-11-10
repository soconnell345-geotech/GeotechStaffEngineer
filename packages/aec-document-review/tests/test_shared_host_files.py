"""One process, several people: what a conversation may reach on the server
(deployment review 2026-11-10).

* A tool result is text, and it can quote a document. A line in an uploaded
  file that read like a save report made the app copy that server file into
  the conversation, as a download.
* The working folder could be any folder on the server: typed into the
  sidebar box, or carried in by a restored zip's ``meta.json``.
* A restored transcript's artifact reference could name any file.

Every name and value here is made up.
"""

import json
import os

from aec_agent_core import _fileio
from aec_document_review import conversation_restore, core
from tests.test_conversation_restore import _zip


def _files_dir(tmp_path, tid="mine"):
    files = tmp_path / "conversations" / tid / "files"
    files.mkdir(parents=True)
    return files


# ---------------------------------------------------------------------------
# Paths a tool result names
# ---------------------------------------------------------------------------

def test_a_reported_path_is_copied_only_from_the_working_or_temporary_folder(
        tmp_path, monkeypatch):
    files = _files_dir(tmp_path)
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(core.tempfile, "gettempdir", lambda: str(temp))
    made = temp / "section.png"
    made.write_bytes(b"png a tool made")
    settings = tmp_path / "app" / ".env"
    settings.parent.mkdir()
    settings.write_text("PROMPTER_API_KEY=made-up\n", encoding="utf-8")
    theirs = _files_dir(tmp_path, "theirs") / "report.pdf"
    theirs.write_bytes(b"%PDF somebody else's")

    copied = core.import_reported_outputs(
        [str(made), str(settings), str(theirs)], str(files))
    assert list(copied) == [os.path.abspath(str(made))]
    assert os.listdir(files) == ["section.png"]


def test_a_custom_working_folder_is_still_a_source(tmp_path, monkeypatch):
    files = _files_dir(tmp_path)
    monkeypatch.setattr(core.tempfile, "gettempdir",
                        lambda: str(tmp_path / "tmp"))
    work = tmp_path / "project_out"
    (work / "figures").mkdir(parents=True)
    (work / "figures" / "plan.png").write_bytes(b"png")
    copied = core.import_reported_outputs(
        [str(work / "figures" / "plan.png")], str(files),
        working_dir=str(work))
    assert len(copied) == 1 and os.listdir(files) == ["plan.png"]


# ---------------------------------------------------------------------------
# The working folder
# ---------------------------------------------------------------------------

def test_a_shared_host_works_in_the_conversations_own_folder(tmp_path):
    root = str(tmp_path)
    elsewhere = tmp_path / "another_person"
    core.set_working_dir("c", str(elsewhere), root=root)
    # one person's own machine: the folder they chose
    assert core.working_dir_for("c", root=root) == \
        os.path.abspath(str(elsewhere))
    _fileio.require_turn_binding(True)          # the proxy named a caller
    assert core.working_dir_for("c", root=root) == \
        core.conversation_files_dir("c", root=root)


def test_a_restored_zip_brings_no_working_folder(tmp_path):
    root = str(tmp_path / "data")
    data = _zip({"transcript.jsonl": '{"role": "user", "text": "hi"}\n'},
                meta={"thread_id": "abc123", "title": "A footing",
                      "working_dir": str(tmp_path / "another_person")})
    res = conversation_restore.restore(data, root=root)
    assert res.status == "restored", res.message
    meta = core.load_meta("abc123", root)
    assert "working_dir" not in meta and meta["title"] == "A footing"
    with open(os.path.join(core.conversation_dir("abc123", root),
                           "meta.json"), encoding="utf-8") as fh:
        assert "working_dir" not in json.load(fh)
    assert core.working_dir_for("abc123", root=root) == \
        core.conversation_files_dir("abc123", root=root)


# ---------------------------------------------------------------------------
# Artifact references in a transcript
# ---------------------------------------------------------------------------

def test_an_artifact_reference_stays_in_the_conversation_folder(tmp_path):
    files = str(_files_dir(tmp_path))
    inside = os.path.join(files, "figures", "plan.png")
    outside = str(tmp_path / "app" / ".env")
    assert core._resolve_artifact("figures/plan.png", files) == \
        os.path.join(files, "figures/plan.png")
    # a relative reference never climbs out, on any host
    assert core._resolve_artifact("../../theirs/files/report.pdf", files) \
        == os.path.join(files, "report.pdf")
    # one person's own machine: an absolute reference is their own file
    assert core._resolve_artifact(outside, files) == outside
    _fileio.require_turn_binding(True)
    assert core._resolve_artifact(outside, files) == \
        os.path.join(files, ".env")
    assert core._resolve_artifact(inside, files) == inside


# ---------------------------------------------------------------------------
# The independent review of the fixes above found three more of the kind
# ---------------------------------------------------------------------------

def test_the_temporary_folder_is_not_a_source_on_a_shared_host(
        tmp_path, monkeypatch):
    """It is everybody's there, and the tools save into the conversation."""
    files = _files_dir(tmp_path)
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(core.tempfile, "gettempdir", lambda: str(temp))
    (temp / "review_memo.docx").write_bytes(b"somebody else's memo")
    _fileio.require_turn_binding(True)
    assert core.import_reported_outputs(
        [str(temp / "review_memo.docx")], str(files)) == {}
    assert os.listdir(files) == []


def test_an_attachment_key_names_a_file_in_the_conversation_only(tmp_path):
    """``attachments.json`` comes with a restored zip: a key was opened
    wherever it pointed, and its bytes handed to the agent by that key."""
    root = str(tmp_path / "data")
    core.ensure_conversation("mine", root=root)
    files = core.conversation_files_dir("mine", root=root)
    with open(os.path.join(files, "plan.pdf"), "wb") as fh:
        fh.write(b"%PDF mine")
    theirs = tmp_path / "theirs.pdf"
    theirs.write_bytes(b"%PDF somebody else's")
    climbing = os.path.relpath(str(theirs), files)
    core.save_attachments_index(
        "mine", ["plan.pdf", climbing, str(theirs)], root=root)
    live = {}
    got = core.load_attachments("mine", live, root=root)
    assert [a.key for a in got] == ["plan.pdf"]
    assert live == {"plan.pdf": b"%PDF mine"}


def test_a_restored_zip_is_a_copy_with_its_own_id_on_a_shared_host(tmp_path):
    """The process finds a conversation's folder and its running turn by id
    alone: a colleague's zip restored under the colleague's id answered into
    the colleague's conversation once both were open."""
    data = _zip({}, meta={"thread_id": "abc123", "title": "A footing",
                          "owner": "dom__alice", "owner_name": "Alice"})
    bob = conversation_restore.restore(
        data, root=str(tmp_path / "bob"), owner="dom__bob", owner_name="Bob")
    carol = conversation_restore.restore(
        data, root=str(tmp_path / "carol"), owner="dom__carol",
        owner_name="Carol")
    assert bob.status == carol.status == "restored"
    assert len({"abc123", bob.thread_id, carol.thread_id}) == 3
    meta = core.load_meta(bob.thread_id, str(tmp_path / "bob"))
    assert meta["thread_id"] == bob.thread_id and meta["owner"] == "dom__bob"
    assert meta["title"] == "A footing"
    # her own zip is a copy too, unless she still has the conversation
    alice_root = str(tmp_path / "alice")
    again = conversation_restore.restore(
        data, root=alice_root, owner="dom__alice", owner_name="Alice")
    assert again.status == "restored" and again.thread_id != "abc123"
    assert core.load_meta(again.thread_id, alice_root)["thread_id"] == \
        again.thread_id


def test_a_conversation_this_person_already_has_keeps_its_id(tmp_path):
    root = str(tmp_path / "alice")
    core.ensure_conversation("abc123", root=root)
    data = _zip({}, meta={"thread_id": "abc123", "title": "A footing",
                          "owner": "dom__alice", "owner_name": "Alice"})
    res = conversation_restore.restore(
        data, root=root, owner="dom__alice", owner_name="Alice")
    assert res.status == "exists" and res.thread_id == "abc123"
    res = conversation_restore.restore(
        data, root=root, owner="dom__alice", owner_name="Alice",
        replace=True)
    assert res.status == "restored" and res.replaced
    assert res.thread_id == "abc123"


def test_one_persons_own_machine_restores_under_the_zips_id(tmp_path):
    res = conversation_restore.restore(
        _zip({}, meta={"thread_id": "abc123", "title": "A footing"}),
        root=str(tmp_path / "solo"))
    assert res.status == "restored" and res.thread_id == "abc123"
