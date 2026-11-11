"""planlens' sheet tools on the Document Review page
(``GEOTECH_REVIEW_SHEET_TOOLS``, BACKLOG PL-37 and DR-07), offline.

Behind the switch the page asks planlens for two tools: ``sheet_index`` (a
drawing set's sheets read from their title blocks) and
``check_cross_references`` (every callout checked against the sheet it
names). Off, or on a planlens that does not have them, the page is the
default build. The last tests run the tools on the coordination tasks' own
drawing set, where every broken callout is known.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.messages import AIMessage  # noqa: E402

from aec_agent_core import document_tools, review_flags  # noqa: E402
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from aec_document_review.review_eval import coordination_fixture as F  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _call, _model, _review_kwargs, _tools_of)

SHEET = ("sheet_index", "check_cross_references")
NAME = "north_yard_annex_drawings_issue_1.pdf"

#: The installed planlens has the sheet tools (after 0.15.1). Document
#: Review pins planlens exactly, so until the pin moves the tests that need
#: them are skipped where the pinned one is installed (CI).
HAVE = document_tools.sheet_tools_available()
needs_sheet_tools = pytest.mark.skipif(
    not HAVE, reason="the installed planlens has no sheet tools")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture
def on(monkeypatch):
    for k, v in review_flags.ARMS["sheet_tools"].items():
        monkeypatch.setenv(k, v)
    assert review_flags.sheet_tools()


def _agent(model, floors=2):
    return build_review_page_agent(
        model, attachments={NAME: F.build_drawing_set(floors, 1).pdf},
        engine=FakeEngine(), **_review_kwargs())


def _last_tool(messages):
    return json.loads(next(m for m in reversed(messages)
                           if getattr(m, "type", "") == "tool").content)


def test_the_names_and_the_arm():
    assert document_tools.SHEET_TOOL_NAMES == SHEET
    assert review_flags.ARMS["sheet_tools"] == {
        "GEOTECH_REVIEW_SHEET_TOOLS": "1"}
    assert "GEOTECH_REVIEW_SHEET_TOOLS" in review_flags.ALL_ENVS
    # planlens publishes them apart from its standing list, so nothing that
    # reads that list offers them unasked
    assert not set(SHEET) & set(document_tools.document_tool_names())


def test_off_the_page_does_not_have_them():
    assert not review_flags.sheet_tools()
    tools = _tools_of(_agent(_model([])))
    assert not set(SHEET) & set(tools)


def test_on_without_them_in_planlens_the_page_is_the_default_build(
        on, monkeypatch):
    """The arm on a planlens that has no sheet tools builds what the
    baseline builds, and says nothing of tools it does not have."""
    monkeypatch.setattr(document_tools, "sheet_tools_available",
                        lambda: False)
    tools = _tools_of(_agent(_model([])))
    assert not set(SHEET) & set(tools)
    assert "open_document" in tools


@needs_sheet_tools
def test_on_the_page_and_its_reading_helper_have_them(on):
    from tests import page_builds
    agent, calls, _ = page_builds.build(
        "review_default", capture=True,
        settings=dict(review_flags.ARMS["sheet_tools"]))
    kw = next(c for c in calls if c["kind"] == "create_deep_agent")["kwargs"]
    by_name = {getattr(t, "name", None): t for t in kw["tools"]}
    assert set(SHEET) <= set(by_name)
    # described in planlens' own words
    for name in SHEET:
        assert by_name[name].description.startswith(
            document_tools.tool_description(name))
    reader = next(s for s in kw["subagents"]
                  if s.get("name") == "general-purpose")
    assert set(SHEET) <= {t.name for t in reader["tools"]}


@needs_sheet_tools
def test_on_the_index_names_every_sheet_of_the_coordination_set(on):
    seen = {}

    def after_open(messages):
        seen["handle"] = _last_tool(messages)["handle"]
        return _call("sheet_index", {"handle": seen["handle"]}, 2)

    def after_index(messages):
        seen["index"] = _last_tool(messages)
        return AIMessage(content="indexed")

    model = _model([_call("open_document", {"source": NAME}),
                    after_open, after_index])
    _agent(model).invoke(
        {"messages": [{"role": "user", "content": "List the sheets."}]},
        config={"recursion_limit": 200})
    built = F.build_drawing_set(2, 1)
    rows = seen["index"]["sheets"]
    assert [(r["sheet"], r["title"]) for r in rows] == list(built.sheets)
    assert [r["pdf_page"] for r in rows] == list(range(1, 16))
    # the index says nothing of A-202: the cover's own list is the agent's
    # to compare with it
    assert "A-202" not in {r["sheet"] for r in rows}


#: What planlens reports for each planted callout: ``unresolved`` where the
#: sheet is in the file and holds no mark for that view; ``not_in_document``
#: where no sheet of that number is in the file (a fact; whether it is a
#: finding depends on the file being the whole set, which the agent judges
#: from the cover's drawing index).
_SMALL = {("7/A-501", "A-101"): "unresolved",
          ("2/A-302", "A-102"): "not_in_document"}
_LONG = {**_SMALL, ("9/A-502", "A-114"): "unresolved"}


@needs_sheet_tools
@pytest.mark.parametrize("floors, planted", [(2, _SMALL), (5, _LONG)])
def test_on_the_check_flags_the_planted_callouts_and_no_others(
        on, floors, planted):
    """The coordination tasks' drawing sets hold exactly these broken
    callouts (``coordination_fixture``): the check must name them all, each
    by what the file shows, and flag no sound one."""
    seen = {}

    def after_open(messages):
        handle = _last_tool(messages)["handle"]
        return _call("check_cross_references", {"handle": handle}, 2)

    def after_check(messages):
        seen["check"] = _last_tool(messages)
        return AIMessage(content="checked")

    model = _model([_call("open_document", {"source": NAME}),
                    after_open, after_check])
    _agent(model, floors).invoke(
        {"messages": [{"role": "user", "content": "Check the callouts."}]},
        config={"recursion_limit": 200})
    result = seen["check"]
    # the flagged rows come first: the callout, the sheet it is drawn on,
    # what the file shows, why, its page and its box
    flagged = [c for c in result["callouts"] if c["status"] != "resolved"]
    assert {(c["callout"], c["sheet"]): c["status"]
            for c in flagged} == planted
    assert result["callouts"][:len(flagged)] == flagged
    built = F.build_drawing_set(floors, 1)
    for c in flagged:
        assert c["why"]
        assert c["page"] == built.page_of(c["sheet"]) - 1
        assert len(c["bbox"]) == 4
    counts = result["counts"]
    assert counts["unresolved"] + counts["not_in_document"] == len(planted)
    assert counts["cannot_tell"] == 0
    assert counts["callouts"] == counts["resolved"] + len(planted)
