"""The notes this page hands its model name no tool the page lacks
(BACKLOG DR-42). A repair.

The note on an upload named ``pdf_import``, ``dxf_import``, ``drawing_ir``
and a ``call_agent`` file_path, and the note after a helper's saves named
``write_diggs``: tools only the geotech page has. Each identifier either
note names must be a tool of the page or an argument of one, checked
against the tools of each build as it is built here (offline, on a fake
model), so the check follows the tools when they change.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("deepagents")

from aec_document_review import core  # noqa: E402
from tests import page_builds  # noqa: E402
from tests.test_review_page_vocabulary_offline import VOCAB  # noqa: E402

#: A word with an underscore in it, or one given a value with "=": how a
#: tool or one of its arguments is named in a note.
IDENT = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b"
                   r"|\b([A-Za-z]\w*)=")

#: Made-up uploads: a report, a drawing, a picture, a spreadsheet.
KEYS = ("report.pdf", "plan.dxf", "photo.png", "takeoff.xlsx")


def _args(tool) -> set:
    schema = tool.tool_call_schema
    if not isinstance(schema, dict):
        schema = schema.model_json_schema()
    return set(schema.get("properties") or {})


@pytest.fixture(scope="module", params=page_builds.REVIEW_BUILDS)
def page_names(request):
    """``(build, tool names, argument names)`` of the page's own agent."""
    agent, _calls, _files = page_builds.build(request.param)
    tools = agent.nodes["tools"].bound.tools_by_name
    args = set().union(*(_args(t) for t in tools.values()))
    return request.param, set(tools), args


def _named(text: str) -> set:
    for key in KEYS:                      # a file's own name is not a tool's
        text = text.replace(f"'{key}'", "''")
    return {a or b for a, b in IDENT.findall(text)}


def _check(text: str, page_names) -> None:
    build, tools, args = page_names
    named = _named(text)
    lacking = sorted(n for n in named if n not in tools | args)
    assert not lacking, f"{build}: the note names {lacking}:\n{text}"
    hits = [m.group(0) for m in VOCAB.finditer(text)]
    assert not hits, f"{build}: {hits}"


def test_the_upload_note_names_only_what_the_page_has(page_names, tmp_path):
    atts = [core.Attachment(key=k, path=str(tmp_path / k), size=1)
            for k in KEYS]
    note = core.attachment_note(atts)
    for key in KEYS:
        assert f"source='{key}'" in note
        assert f"attachment_key='{key}'" in note
        assert f"the file '{key}' in the working folder" in note
    assert str(tmp_path) not in note      # never the server path (A6)
    assert {"source", "attachment_key"} <= _named(note)
    _check(note, page_names)


def test_the_note_after_a_helpers_saves_names_only_what_the_page_has(
        page_names):
    from aec_agent_core.deep import keep_work
    from aec_agent_core.hooks import get_hook
    # loading the package fills the hook with this page's wording
    assert get_hook(keep_work.SAVED_FILES_HINT_HOOK) is core.saved_files_hint
    note = keep_work.saved_note(["takeoff.xlsx"])
    assert note == ("[Files this helper saved, in the working folder: "
                    "'takeoff.xlsx'. " + core.SAVED_FILES_HINT + "]")
    assert "write_diggs" not in note
    _check(note, page_names)
