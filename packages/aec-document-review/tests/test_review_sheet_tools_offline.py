"""planlens' sheet tools on the Document Review page
(``GEOTECH_REVIEW_SHEET_TOOLS``, BACKLOG PL-37 and DR-07), offline.

Behind the switch the page asks planlens for two tools: ``sheet_index`` (a
drawing set's sheets read from their title blocks) and
``check_cross_references`` (every callout checked against the sheet it
names); and, where the installed planlens has it, ``check_schedule_marks``
(the set's schedules checked against the marks on its sheets, BACKLOG
PL-42), with ``open_document`` bringing the sheet index and that check.
Off, or on a planlens that does not have them, the page is the default
build. The tests run the tools on the coordination tasks' own drawing set,
where every broken callout and every planted mark problem is known.
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


# -- the schedule check (planlens after 0.15.1 with PL-42) -------------------

MARKS = "check_schedule_marks"
AT_OPEN = ("sheet_index", "schedule_marks")

#: The installed planlens checks the marks against the schedules too.
HAVE_MARKS = HAVE and document_tools.has_tool(MARKS)
needs_schedule_marks = pytest.mark.skipif(
    not HAVE_MARKS, reason="the installed planlens has no schedule check")

#: What the check says of each planted mark problem of the coordination
#: sets (``coordination_fixture``), by its key: a mark drawn with no row is
#: ``no_row``, a row drawn on no sheet ``no_mark``.
_MARK_KEYS = {
    "door-not-scheduled": ("109", "no_row"),
    "door-not-on-plan": ("204", "no_mark"),
    "footing-not-scheduled": ("F4", "no_row"),
    "door-not-scheduled-4": ("407", "no_row"),
    "door-not-on-plan-5": ("509", "no_mark"),
    "fcu-not-scheduled": ("FCU-3-4", "no_row"),
    "panel-not-scheduled": ("LP-4B", "no_row"),
}


def _opened_and_checked(floors):
    """open_document, then check_schedule_marks, on the coordination set:
    both results."""
    seen = {}

    def after_open(messages):
        seen["open"] = _last_tool(messages)
        return _call(MARKS, {"handle": seen["open"]["handle"]}, 2)

    def after_check(messages):
        seen["check"] = _last_tool(messages)
        return AIMessage(content="checked")

    model = _model([_call("open_document", {"source": NAME}),
                    after_open, after_check])
    _agent(model, floors).invoke(
        {"messages": [{"role": "user", "content": "Check the schedules."}]},
        config={"recursion_limit": 200})
    return seen["open"], seen["check"]


def _opened(floors=2):
    seen = {}

    def after_open(messages):
        seen["open"] = _last_tool(messages)
        return AIMessage(content="opened")

    model = _model([_call("open_document", {"source": NAME}), after_open])
    _agent(model, floors).invoke(
        {"messages": [{"role": "user", "content": "Open it."}]},
        config={"recursion_limit": 200})
    return seen["open"]


def test_the_schedule_check_is_an_optional_sheet_tool():
    assert document_tools.OPTIONAL_SHEET_TOOL_NAMES == (MARKS,)
    assert not set(document_tools.OPTIONAL_SHEET_TOOL_NAMES) & set(
        document_tools.document_tool_names())
    names = document_tools.sheet_tool_names()
    if HAVE:
        assert names[:2] == SHEET
        assert (MARKS in names) is HAVE_MARKS
    else:
        assert names == ()


def test_off_open_document_brings_neither_and_the_check_is_absent():
    assert not review_flags.sheet_tools()
    assert MARKS not in _tools_of(_agent(_model([])))
    opened = _opened()
    assert not set(AT_OPEN) & set(opened)


def test_on_with_a_planlens_without_the_check_the_page_has_the_two(
        on, monkeypatch):
    """A planlens with the two sheet tools and no schedule check (0.15.x
    after PL-37): the page offers the two, and nothing of the third."""
    if not HAVE:
        pytest.skip("the installed planlens has no sheet tools")
    real = document_tools.has_tool
    monkeypatch.setattr(document_tools, "has_tool",
                        lambda name: name != MARKS and real(name))
    tools = _tools_of(_agent(_model([])))
    assert set(SHEET) <= set(tools)
    assert MARKS not in tools


@needs_schedule_marks
def test_on_the_page_and_its_reading_helper_have_the_check(on):
    from tests import page_builds
    agent, calls, _ = page_builds.build(
        "review_default", capture=True,
        settings=dict(review_flags.ARMS["sheet_tools"]))
    kw = next(c for c in calls if c["kind"] == "create_deep_agent")["kwargs"]
    by_name = {getattr(t, "name", None): t for t in kw["tools"]}
    assert by_name[MARKS].description.startswith(
        document_tools.tool_description(MARKS))
    reader = next(s for s in kw["subagents"]
                  if s.get("name") == "general-purpose")
    assert MARKS in {t.name for t in reader["tools"]}


@needs_schedule_marks
@pytest.mark.parametrize("floors", [2, 5])
def test_on_the_check_flags_the_planted_mark_problems_and_no_others(
        on, floors):
    """The coordination sets' planted mark problems (a door, a footing, a
    fan coil unit or a panel drawn with no row; a door scheduled and drawn
    nowhere): open_document brings them with the sheet index, and the
    check names them all, each on its sheet with both page numbers, and
    nothing else."""
    built = F.build_drawing_set(floors, 1)
    planted = {_MARK_KEYS[p.key] for p in built.planted
               if p.key in _MARK_KEYS}
    assert len(planted) == (3 if floors == 2 else 7)
    opened, check = _opened_and_checked(floors)
    # what open_document brings: every sheet, and the check's first items
    index = opened["sheet_index"]
    assert [(r["sheet"], r["title"]) for r in index["sheets"]] == \
        list(built.sheets)
    at_open = opened["schedule_marks"]
    assert {(i["mark"], i["status"]) for i in at_open["items"]} == planted
    assert at_open["counts"]["schedules"] == (2 if floors == 2 else 4)
    # the check itself
    items = check["items"]
    assert {(i["mark"], i["status"]) for i in items} == planted
    for item in items:
        assert item["pdf_page"] == item["page"] + 1
        assert item["schedule"] and item["why"] and len(item["bbox"]) == 4
        if item["side"] == "mark":
            assert item["page"] == built.page_of(item["sheet"]) - 1
    door = next(i for i in items if i["mark"] == "109")
    assert (door["sheet"], door["pdf_page"]) == ("A-101", 4)
    assert door["schedule"].startswith("DOOR SCHEDULE (A-601")
    assert check["counts"]["cannot_tell"] == 0
    assert check["counts"]["may_be_on_unread_sheet"] == 0


@needs_schedule_marks
def test_on_a_scanned_set_the_check_says_it_read_no_sheet(on):
    seen = {}

    def after_open(messages):
        seen["open"] = _last_tool(messages)
        return AIMessage(content="opened")

    model = _model([_call("open_document", {"source": "scan.pdf"}),
                    after_open])
    agent = build_review_page_agent(
        model, attachments={"scan.pdf": F.build_scanned_set()},
        engine=FakeEngine(), **_review_kwargs())
    agent.invoke({"messages": [{"role": "user", "content": "Open it."}]},
                 config={"recursion_limit": 200})
    marks = seen["open"]["schedule_marks"]
    assert marks["counts"]["schedules"] == 0
    assert "items" not in marks
    assert marks["pages_unread"] == "0-14"
    assert "15 sheet(s) could not be read" in marks["note"]
