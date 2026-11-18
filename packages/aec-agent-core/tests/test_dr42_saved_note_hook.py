"""The note after a helper's saves: the sentence after the file names is the
app's (BACKLOG DR-42). A repair.

The core's note told every page "write_diggs reads records from
investigations_file / lab_tests_file", a tool only the geotech page has.
An app now words that sentence through the hook
``keep_work.saved_files_hint``; unfilled, the note is exactly what the
geotech page has been given since RI-31.
"""

from __future__ import annotations

import pytest

from aec_agent_core import hooks
from aec_agent_core.deep import keep_work

#: The note as the geotech page has been given it since RI-31, byte for byte.
GEOTECH_NOTE = (
    "[Files this helper saved, in the working folder: 'b1.json'. Give a "
    "file to the next tool by its name (write_diggs reads records from "
    "investigations_file / lab_tests_file) -- never retype what it holds.]")


@pytest.fixture(autouse=True)
def empty_hooks():
    with hooks._LOCK:
        saved = (dict(hooks._HOOKS), dict(hooks._RESOLVED))
        hooks._HOOKS.clear()
        hooks._RESOLVED.clear()
    yield
    with hooks._LOCK:
        hooks._HOOKS.clear()
        hooks._RESOLVED.clear()
        hooks._HOOKS.update(saved[0])
        hooks._RESOLVED.update(saved[1])


def test_unfilled_the_note_is_the_geotech_wording_byte_for_byte():
    assert keep_work.saved_note(["b1.json"]) == GEOTECH_NOTE
    many = keep_work.saved_note([f"f{i}.json" for i in range(12)])
    assert many.startswith("[Files this helper saved, in the working "
                           "folder: 'f0.json', ")
    assert "'f9.json' and 2 more. Give a file to the next tool by its " \
           "name (write_diggs reads" in many and "'f10.json'" not in many


def test_an_app_words_the_sentence_after_the_names():
    seen = []
    hooks.set_hook(keep_work.SAVED_FILES_HINT_HOOK,
                   lambda names: seen.append(names) or "Open them by name.")
    note = keep_work.saved_note(["takeoff.xlsx", "notes.md"])
    assert note == ("[Files this helper saved, in the working folder: "
                    "'takeoff.xlsx', 'notes.md'. Open them by name.]")
    assert seen == [["takeoff.xlsx", "notes.md"]]
    assert "write_diggs" not in note


@pytest.mark.parametrize("answer", ["", "   ", None, 7, RuntimeError])
def test_a_hook_that_fails_or_says_nothing_keeps_the_default(answer):
    def hint(_names):
        if answer is RuntimeError:
            raise RuntimeError("boom")
        return answer
    hooks.set_hook(keep_work.SAVED_FILES_HINT_HOOK, hint)
    assert keep_work.saved_note(["b1.json"]) == GEOTECH_NOTE


def test_a_hook_that_will_not_import_keeps_the_default():
    hooks.set_hook(keep_work.SAVED_FILES_HINT_HOOK,
                   "no_such_module_dr42:hint")
    assert keep_work.saved_note(["b1.json"]) == GEOTECH_NOTE
