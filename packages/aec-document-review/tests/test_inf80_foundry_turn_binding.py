"""Document Review on Foundry with several people at once (BACKLOG INF-80).

A published Foundry app is one process for everyone who opens it, and no
sign-in header names them. Until INF-80 the page's protections for a shared
process (each turn's tools bound to its own conversation's folder, the
working folder always the conversation's own, the **Working folder** box
hidden) switched on only when a header named the caller, so on Foundry they
were off. ``identity.current_identity`` now turns the per-turn binding on
for a Foundry deployment (``GEOTECH_DEPLOYMENT=foundry``) without naming
anyone: the conversation list stays shared (OWN-152) and nobody is refused.

The fake dataset library and the env cleaning come from
``tests/test_foundry_host.py``. Every name, path and value here is made up.
"""

import os
import threading

import pytest

import aec_document_review.core as core
import aec_document_review.sharepoint_store as sp
from aec_agent_core import _fileio, engine_config
from aec_agent_core.engine_config import EngineResolution
from aec_document_review import foundry_entry, identity, profiles
from tests.test_foundry_host import (  # noqa: F401 - fixtures used by name
    _clean, _conversation, _foundry, dataset)

BOX_LABEL = "Working folder (agent saves land here)"


@pytest.fixture(autouse=True)
def _binding_off_at_the_start(monkeypatch):
    """Each test starts as a fresh process does: no turn binding required
    (conftest puts the flag back afterwards)."""
    monkeypatch.setattr(_fileio, "_REQUIRE_BINDING", False)
    monkeypatch.delenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, raising=False)


def _start_page_on_foundry(monkeypatch, how):
    if how == "the Foundry entry":
        foundry_entry.mark_deployment()
    else:                                  # a workspace that sets it itself
        monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    return identity.current_identity()     # what app.py does on every run


# ---------------------------------------------------------------------------
# binding required, nobody named
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("how", ["the Foundry entry", "another start"])
def test_on_foundry_with_no_header_each_turn_must_be_bound(
        monkeypatch, tmp_path, how):
    who = _start_page_on_foundry(monkeypatch, how)
    assert who is identity.ANONYMOUS and not who.multi_user
    assert _fileio.turn_binding_required()
    assert core._shared_host()
    # A context with no folder bound refuses, rather than use the
    # process-wide folder, which names whoever reran last: here another
    # conversation's.
    theirs = tmp_path / "conversations" / "made-up-theirs" / "files"
    theirs.mkdir(parents=True)
    monkeypatch.setenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, str(theirs))
    with pytest.raises(_fileio.UnboundWorkingDir):
        _fileio.host_output_dir()
    # and the page no longer points that variable at a conversation
    core.apply_default_output_dir(str(theirs))
    assert _fileio.DEFAULT_OUTPUT_DIR_ENV not in os.environ


def test_on_foundry_a_launcher_email_still_binds_each_turn(monkeypatch):
    """A name in the environment names nobody on Foundry: one process still
    serves everyone who opens the app."""
    monkeypatch.setenv("GEOTECH_USER_EMAIL", "made.up@example.test")
    who = _start_page_on_foundry(monkeypatch, "the Foundry entry")
    assert who.source == "email" and not who.multi_user
    assert _fileio.turn_binding_required()


@pytest.mark.parametrize("deployment", [None, "made-up-host"])
def test_off_foundry_with_no_header_nothing_changes(monkeypatch, deployment):
    if deployment:
        monkeypatch.setenv("GEOTECH_DEPLOYMENT", deployment)
    who = identity.current_identity()
    assert who is identity.ANONYMOUS
    assert not _fileio.turn_binding_required()
    assert not core._shared_host()


def test_on_foundry_nobody_is_refused_and_the_list_stays_shared(
        monkeypatch):
    who = _start_page_on_foundry(monkeypatch, "the Foundry entry")
    assert _fileio.turn_binding_required()
    assert not who.refused
    assert not who.multi_user
    # one shared folder of conversations, not one per viewer
    assert profiles.session_root(profiles.DOCUMENT_REVIEW, who) == \
        os.path.join(core.data_root(), "pages", profiles.DOCUMENT_REVIEW.name)


# ---------------------------------------------------------------------------
# the working folder
# ---------------------------------------------------------------------------

def test_on_foundry_the_working_folder_is_the_conversations_own(
        monkeypatch, tmp_path):
    """Even when its meta names another folder: typed into the box before
    this change by anyone who can open the app, or carried in by a restore."""
    root = str(tmp_path / "pages")
    theirs = core.conversation_files_dir("made-up-theirs", root)
    core.ensure_conversation("made-up-mine", root=root)
    meta = core.load_meta("made-up-mine", root)
    meta["working_dir"] = theirs
    core.save_meta("made-up-mine", meta, root)
    _start_page_on_foundry(monkeypatch, "the Foundry entry")
    assert core.working_dir_for("made-up-mine", root=root) == \
        core.conversation_files_dir("made-up-mine", root)


def test_on_foundry_the_working_folder_box_is_not_drawn(
        dataset, monkeypatch):
    """The box's condition (``app.py``) is off, and the page Foundry
    publishes, run by Streamlit, draws no box and raises nothing."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    _foundry(monkeypatch)
    who = identity.current_identity()
    assert not (not who.multi_user and not core._shared_host())
    monkeypatch.setattr(_fileio, "_REQUIRE_BINDING", False)   # page sets it
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
    assert _fileio.turn_binding_required()
    labels = [str(t.label) for t in at.text_input]
    assert BOX_LABEL not in labels, labels
    assert at.chat_input
    assert not any("could not tell who you are" in str(e.value)
                   for e in at.error)


# ---------------------------------------------------------------------------
# two people's turns at the same moment
# ---------------------------------------------------------------------------

def test_two_turns_at_once_on_foundry_each_save_into_their_own_conversation(
        monkeypatch, tmp_path):
    """Two viewers each pointed their conversation at one project folder
    with the box (which anyone could, before this change). Their turns run
    at once, each started as the page starts one (``app.py``: the working
    folder, the process-wide variable, a snapshot; ``turn_jobs``: the folder
    bound in the worker thread), and each saves ``answer.txt``. Each
    conversation must end with its own answer only."""
    root = str(tmp_path / "pages")
    shared = tmp_path / "made_up_project_out"
    shared.mkdir()
    for tid in ("made-up-a", "made-up-b"):
        core.set_working_dir(tid, str(shared), root=root)
    _start_page_on_foundry(monkeypatch, "the Foundry entry")

    a_saved, b_started, b_done = (threading.Event(), threading.Event(),
                                  threading.Event())
    results, errors = {}, []

    def turn(tid, text):
        # the page, at turn start
        files = core.conversation_files_dir(tid, root)
        wd = core.working_dir_for(tid, root=root)
        core.apply_default_output_dir(wd)
        before_wd = (core.snapshot_dir(wd)
                     if os.path.abspath(wd) != os.path.abspath(files)
                     else None)

        def worker():
            try:
                _fileio.bind_working_dir(wd)             # turn_jobs
                if tid == "made-up-a":
                    b_started.wait(5)
                else:
                    b_started.set()
                    a_saved.wait(5)                  # A saves first, then B
                target = _fileio.resolve_output_path("answer.txt")
                with open(target, "w", encoding="utf-8") as fh:
                    fh.write(text)
                if tid == "made-up-a":
                    a_saved.set()
                    b_done.wait(5)                   # both saved, then
                else:
                    b_done.set()
                if before_wd is not None:            # the post-turn import
                    core.import_external_artifacts(wd, files, before_wd, [])
                results[tid] = files
            except Exception as exc:                 # noqa: BLE001
                errors.append(exc)
                a_saved.set()
                b_done.set()

        t = threading.Thread(target=worker)
        t.start()
        return t

    threads = [turn("made-up-a", "made-up answer A"),
               turn("made-up-b", "made-up answer B")]
    for t in threads:
        t.join(10)
    assert not errors, errors
    for tid, text in (("made-up-a", "made-up answer A"),
                      ("made-up-b", "made-up answer B")):
        files = results[tid]
        assert os.listdir(files) == ["answer.txt"], (tid, os.listdir(files))
        with open(os.path.join(files, "answer.txt"), encoding="utf-8") as fh:
            assert fh.read() == text, tid
    assert os.listdir(shared) == []


# ---------------------------------------------------------------------------
# the Foundry dataset store needs no turn's folder
# ---------------------------------------------------------------------------

def test_on_foundry_background_saves_and_restores_work_with_binding_on(
        dataset, monkeypatch, tmp_path):
    """The store's background save runs on a thread of its own, with no
    folder bound; it and a restore name every path themselves."""
    _foundry(monkeypatch)
    identity.current_identity()
    assert _fileio.turn_binding_required()
    store = sp.SharePointStore()
    assert store.backend == "foundry" and store.background
    root = str(tmp_path / "container")
    _conversation(root, "T-INF80", "Made-up drawing check")
    store.save_soon("T-INF80", root)
    assert store.flush(30)
    assert any(p.startswith("Document Review/conversations/")
               and p.endswith("/transcript.jsonl") for p in dataset.files_)
    names = [r["name"] for r in store.list_remote_conversations()]
    assert len(names) == 1, names
    got = {}

    def restore():                     # a thread with no folder bound
        got.update(store.restore_conversation(
            names[0], root=str(tmp_path / "fresh")))

    t = threading.Thread(target=restore)
    t.start()
    t.join(30)
    assert got.get("status") == "restored", got
    assert got.get("thread_id") == "T-INF80"
