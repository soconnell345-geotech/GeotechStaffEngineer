"""The note after a save names the file and rules out dead links
(BACKLOG DR-43 part 1). A repair.

Foundry brief 7 (TRACE_REVIEW F-11): 15 runs on one model linked a file
they had written as ``sandbox:/mnt/data/...``, which reaches no one. The
note ``method_call`` gives its writers said "a /tmp path or a sandbox: link
would not reach them"; the note after ``write_docx`` and the other saves
(``core.ATTACHMENT_NOTE``, the save function's ``saved_note``) did not.
"""

from __future__ import annotations

import os

from aec_document_review import core

LINE = ("Tell the user the file name; a /tmp path or a sandbox: link would "
        "not reach them.")


def test_the_attachment_note_names_the_file_and_rules_out_dead_links():
    assert LINE in core.ATTACHMENT_NOTE
    assert core.ATTACHMENT_NOTE.startswith("This file is now ATTACHED")


def test_every_save_into_the_conversation_carries_it(tmp_path):
    files = str(tmp_path)
    save = core.make_save_fn(files, [])
    saved = save("schedule.docx", b"PK\x03\x04")
    assert os.path.isfile(saved)
    assert LINE in save.saved_note(saved)
    # a save outside the conversation's files gets no note, as before
    assert save.saved_note(str(tmp_path.parent / "elsewhere.docx")) is None
