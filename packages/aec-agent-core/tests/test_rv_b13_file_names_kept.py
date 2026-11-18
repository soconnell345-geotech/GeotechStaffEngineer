"""The release review of the whole change after 1.2.0 (finding 1): the
masking of a tool's error line took a document's file name for a key.

``error_line`` did not mask at 1.2.0. Since the record masks, a tool error
that names a file whose name starts ``sk-`` / ``pk-`` / ``rk-`` (a sketch
sheet saved as ``sk-101-foundation-plan.pdf``) reached the model as
``'….pdf'``: a tool result that no longer says which file it could not
open, with every setting unset. A file name is kept; a key is still masked.
All values are made up.
"""

import pytest

from aec_agent_core import error_text

FILES = ("sk-101-foundation-plan.pdf", "pk-204-parking-level.dxf",
         "rk-300-rock-anchors.DOCX", "sk-101-foundation-plan.jpeg")


@pytest.mark.parametrize("name", FILES)
def test_a_file_name_is_kept_in_an_error_text(name):
    text = f"no document named '{name}' among the attachments"
    assert error_text.mask_error_secrets(text) == text


def test_a_tool_error_still_names_the_file_it_could_not_open():
    exc = ValueError("no document named 'sk-101-foundation-plan.pdf'")
    out = error_text.tool_error("open_document", exc)
    assert "sk-101-foundation-plan.pdf" in out["error"]
    assert "…" not in out["error"]


@pytest.mark.parametrize("text, gone", [
    ("Incorrect API key provided: sk-fake-00000000-not-real.", "sk-fake"),
    ("refused: sk-fake-00000000-not-real", "sk-fake"),
    ("key sk-fake-00000000-not-real, then nothing", "sk-fake"),
    # a full stop and a new sentence is not a file extension
    ("it was pk-fake-00000000-not-real.Another line", "pk-fake"),
    # nor is a longer word after the stop
    ("it was rk-fake-00000000-not-real.pdfx", "rk-fake"),
])
def test_a_key_is_still_masked(text, gone):
    out = error_text.mask_error_secrets(text)
    assert gone not in out and "00000000" not in out
    assert "…" in out
