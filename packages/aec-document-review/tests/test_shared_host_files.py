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
