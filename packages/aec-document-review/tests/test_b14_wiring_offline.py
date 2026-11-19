"""Three planlens parts on the Document Review page, each off unless asked
(brief 14), offline.

* Sheet numbers as addresses (BACKLOG PL-37), under the existing
  ``GEOTECH_REVIEW_SHEET_TOOLS``: where the installed planlens offers them
  (``planlens.review.specs.sheet_args``), a document tool that takes
  ``pages`` also takes ``sheets`` and one that takes ``page`` also takes
  ``sheet``.
* ``drawing_geometry`` (BACKLOG DR-44, PL-13), behind
  ``GEOTECH_REVIEW_DRAWING_TOOLS``.
* A DXF also as a PDF plot of its model space (BACKLOG DR-19), behind
  ``GEOTECH_REVIEW_DXF_PLOTS``.

With every switch unset the page hands the model what it did before (a
hash recorded on the commit before this work), and on a planlens without
these parts (0.16.0) every switch set changes nothing but what the sheet
tools' switch already gave. Every document, sheet number and drawing here
is made up in the test or by planlens' own fixtures.
"""

import hashlib
import inspect
import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core import document_tools, review_flags  # noqa: E402
from aec_agent_core.deep import vision_kit  # noqa: E402
from tests.test_review_harness_offline import FakeEngine  # noqa: E402

SHEETS_ENV = "GEOTECH_REVIEW_SHEET_TOOLS"
DRAWING_ENV = "GEOTECH_REVIEW_DRAWING_TOOLS"
DXF_ENV = "GEOTECH_REVIEW_DXF_PLOTS"
NEW_ENVS = (DRAWING_ENV, DXF_ENV)
DRAWING = "drawing_geometry"
SET = "set.pdf"
DXF = "site.dxf"

#: The page's document tools that take ``pages`` (planlens' own list of
#: them is its ``_ARGS``; ``find_like`` reads an attachment, with no handle
#: whose sheet index could say where a sheet is).
TAKE_SHEETS = ("render_page_thumbnails", "document_page_map",
               "read_document", "search_document", "find_quantities",
               "document_markups", "check_cross_references",
               "check_schedule_marks")

#: What the installed planlens has of the three parts.
HAVE_SHEET_ADDRESSES = bool(document_tools.sheet_address_args(
    "read_document"))
HAVE_DRAWING = document_tools.drawing_tools_available()
HAVE_DXF_PLOTS = document_tools.dxf_plots_available()
needs_sheet_addresses = pytest.mark.skipif(
    not HAVE_SHEET_ADDRESSES,
    reason="the installed planlens takes no sheet numbers (0.16.0)")
needs_drawing = pytest.mark.skipif(
    not HAVE_DRAWING,
    reason="the installed planlens has no drawing tools (0.16.0)")
needs_dxf_plots = pytest.mark.skipif(
    not HAVE_DXF_PLOTS,
    reason="the installed planlens plots no DXF (0.16.0)")

#: The page's document tools and its two page looks, as the model is shown
#: them with every switch unset: sha256 of :func:`_surface`, recorded on
#: the commit this work started from (69d6c2c) with planlens 0.16.0 and
#: with the planlens that has the three parts (the two agree).
UNSET_SURFACE = (
    "30d854b340895a7723dfaddc1dd9c20e13f1c7892f815330447034eefcf54c63")
#: ``tests.page_builds.planlens_surface()`` of the planlens it was recorded
#: with: the hash holds planlens' own words, so it is compared only there.
RECORDED_PLANLENS = ("3e415249e4551735",)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.LATER_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    from planlens.review import conversation
    conversation._SPACES.clear()


@pytest.fixture(scope="module")
def sheet_set():
    from planlens.testing.sheet_set_fixtures import build_sheet_set
    return build_sheet_set()


@pytest.fixture(scope="module")
def dxf(tmp_path_factory):
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (60, 0), (60, 30), (0, 30)], close=True)
    msp.add_line((0, 15), (60, 15))
    msp.add_text("GRID B", height=2.0).set_placement((4, 22))
    path = tmp_path_factory.mktemp("b14") / DXF
    doc.saveas(str(path))
    return path.read_bytes()


def _kit(attachments, **kw):
    return {t.name: t for t in vision_kit.make_vision_tools(
        engine=FakeEngine(), attachments=attachments, **kw)}


def _schema(tool):
    schema = tool.tool_call_schema
    if not isinstance(schema, dict):
        schema = schema.model_json_schema()
    return schema


def _shown(tool):
    return {"description": inspect.cleandoc(tool.description),
            "input_schema": _schema(tool)}


def _all_shown(kit):
    return {name: _shown(tool) for name, tool in kit.items()}


def _page(**settings):
    """``(tools by name, the reading helper's tool names)`` of the page
    built under ``settings``."""
    from tests import page_builds
    _agent, calls, _ = page_builds.build("review_default", capture=True,
                                         settings=settings)
    kw = next(c for c in calls if c["kind"] == "create_deep_agent")["kwargs"]
    reader = next(s for s in kw["subagents"]
                  if s.get("name") == "general-purpose")
    return ({t.name: t for t in kw["tools"] if hasattr(t, "name")},
            [t.name for t in reader["tools"]])


def _surface(tools) -> str:
    names = sorted((set(document_tools.document_tool_names())
                    | {"analyze_pdf_page", "render_region"}) & set(tools))
    shown = {n: {"description": tools[n].description,
                 "schema": _schema(tools[n])} for n in names}
    return hashlib.sha256(json.dumps(shown, sort_keys=True).encode(
        "utf-8")).hexdigest()


def _call(kit, name, **args):
    return json.loads(kit[name].invoke(args))


# -- the two new switches -------------------------------------------------------

@pytest.mark.parametrize("env, read", [
    (DRAWING_ENV, "drawing_tools"), (DXF_ENV, "dxf_plots")])
def test_each_new_switch_is_a_later_one_and_off_unless_set(
        monkeypatch, env, read):
    is_on = getattr(review_flags, read)
    assert env in review_flags.LATER_ENVS
    # the geotech app's tests pin these two whole
    assert env not in review_flags.ALL_ENVS
    assert not any(env in arm for arm in review_flags.ARMS.values())
    assert f"``{env}`` (default off)" in review_flags.__doc__
    assert not is_on()
    assert env not in review_flags.describe()
    for off in ("0", "false", "no", ""):
        monkeypatch.setenv(env, off)
        assert not is_on()
    monkeypatch.setenv(env, "1")
    assert is_on()
    assert f"{env}=1" in review_flags.describe()
    with review_flags.switches({}):
        assert not is_on()
    with review_flags.switches({env: "1"}):
        assert is_on()


def test_the_readme_lists_both_switches():
    import os
    readme = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "README.md")
    with open(readme, encoding="utf-8") as fh:
        text = fh.read()
    for env in NEW_ENVS:
        assert f"| `{env}` |" in text


# -- every switch unset: what it was --------------------------------------------

def test_unset_the_page_shows_what_it_did_before_this_work():
    from tests import page_builds
    tools, reader = _page()
    assert DRAWING not in tools and DRAWING not in reader
    for tool in tools.values():
        assert not {"sheets", "sheet"} & (
            set(_schema(tool)["properties"]) - _own_sheets(tool))
    if page_builds.planlens_surface() not in RECORDED_PLANLENS:
        pytest.skip("another planlens than the hash was recorded with: the "
                    "fingerprint tests are the comparison there")
    assert _surface(tools) == UNSET_SURFACE


def _own_sheets(tool):
    """``write_xlsx`` has an argument of its own called ``sheets`` (a
    workbook's)."""
    return {"sheets"} if tool.name == "write_xlsx" else set()


def test_unset_the_kit_and_the_host_are_the_default(sheet_set):
    att = {SET: sheet_set.pdf}
    plain = _kit(att)
    assert DRAWING not in plain
    asked_off = _kit(att, drawing_tools=False, dxf_plots=False)
    assert _all_shown(plain) == _all_shown(asked_off)
    options = document_tools.host_for().toolkit_options or {}
    assert not {"drawing_tools", "dxf_plots"} & set(options)
    # a sheet number is an argument the tools do not take
    refused = plain["read_document"].invoke({"handle": SET,
                                             "sheets": "A-101"})
    assert "sheets" in refused and "sheet_pages" not in refused


def test_unset_a_dxf_is_text_and_its_page_tools_refuse_it(dxf):
    kit = _kit({DXF: dxf})
    opened = _call(kit, "open_document", source=DXF)
    assert "plot" not in opened and "GRID B" in opened["text"]
    refused = _call(kit, "document_page_map", handle=DXF)
    assert refused["error"].startswith(f"'{DXF}' is a DXF drawing")
    assert "do not apply to it." in kit["open_document"].description


# -- a planlens without the parts: every switch set is the build it was --------

def test_set_without_the_parts_in_planlens_nothing_is_added(
        sheet_set, dxf, monkeypatch):
    monkeypatch.setattr(document_tools, "sheet_address_args",
                        lambda name: [])
    monkeypatch.setattr(document_tools, "drawing_tools_available",
                        lambda: False)
    monkeypatch.setattr(document_tools, "dxf_plots_available",
                        lambda: False)
    att = {SET: sheet_set.pdf, DXF: dxf}
    before = _kit(att, sheet_tools=True)
    asked = _kit(att, sheet_tools=True, drawing_tools=True, dxf_plots=True)
    assert _all_shown(asked) == _all_shown(before)
    assert DRAWING not in asked
    options = document_tools.host_for(
        sheet_tools=True, drawing_tools=True,
        dxf_plots=True).toolkit_options or {}
    assert options == {"sheet_tools": True}
    refused = _call(asked, "document_page_map", handle=DXF)
    assert refused["error"].startswith(f"'{DXF}' is a DXF drawing")


@pytest.mark.skipif(HAVE_SHEET_ADDRESSES or HAVE_DRAWING or HAVE_DXF_PLOTS,
                    reason="the installed planlens has the parts")
def test_set_on_planlens_0_16_the_page_is_the_build_it_was(sheet_set, dxf):
    """On the released planlens, which has none of the three, the two new
    switches give the default page and the sheet tools' switch gives what
    it gave; nothing fails."""
    unset_tools, unset_reader = _page()
    new_tools, new_reader = _page(**{env: "1" for env in NEW_ENVS})
    assert _all_shown(new_tools) == _all_shown(unset_tools)
    assert new_reader == unset_reader
    sheet_tools, sheet_reader = _page(**{SHEETS_ENV: "1"})
    all_tools, all_reader = _page(**{SHEETS_ENV: "1", DRAWING_ENV: "1",
                                     DXF_ENV: "1"})
    assert _all_shown(all_tools) == _all_shown(sheet_tools)
    assert all_reader == sheet_reader
    att = {SET: sheet_set.pdf, DXF: dxf}
    kit = _kit(att, sheet_tools=True, drawing_tools=True, dxf_plots=True)
    assert "error" not in _call(kit, "open_document", source=SET)
    assert "plot" not in _call(kit, "open_document", source=DXF)
    assert "sheets" in kit["read_document"].invoke(
        {"handle": SET, "sheets": "A-101"})        # refused by name


# -- sheet numbers as addresses, under the sheet tools' switch ------------------

@needs_sheet_addresses
def test_sheet_numbers_are_not_offered_without_the_sheet_tools(sheet_set):
    kit = _kit({SET: sheet_set.pdf}, plan_tools=True, drawing_tools=True)
    for name, tool in kit.items():
        assert not {"sheets", "sheet"} & (
            set(_schema(tool)["properties"]) - _own_sheets(tool)), name
    plain = _kit({SET: sheet_set.pdf})
    assert _shown(kit["read_document"]) == _shown(plain["read_document"])


@needs_sheet_addresses
def test_with_the_sheet_tools_each_page_argument_has_its_sheet_argument(
        sheet_set):
    from planlens.review import specs as host_specs
    kit = _kit({SET: sheet_set.pdf}, sheet_tools=True, plan_tools=True,
               drawing_tools=True)
    plain = _kit({SET: sheet_set.pdf})
    took = set()
    for name, tool in kit.items():
        props = _schema(tool)["properties"]
        added = [a for a in ("sheets", "sheet")
                 if a in set(props) - _own_sheets(tool)]
        theirs = host_specs.sheet_args(name, True)
        # exactly what planlens offers for the tool, in its schema and
        # words, after the tool's other arguments
        assert added == [arg for arg, _schema_ in theirs], name
        for arg, schema in theirs:
            assert props[arg] == dict(schema, title=arg.title()), name
            assert list(props)[-len(theirs):] == [a for a, _s in theirs]
        if added:
            took.add(name)
        if name in plain:
            # nothing else about the tool moved
            before = _shown(plain[name])
            now = _shown(tool)
            for arg in added:
                now["input_schema"]["properties"].pop(arg)
            assert now == before, name
    assert set(TAKE_SHEETS) <= took
    assert {"plan_symbols", "plan_grid"} <= took
    if HAVE_DRAWING:
        assert DRAWING in took
    # the looks and the readers that are the app's own take none
    assert not {"analyze_pdf_page", "render_region", "read_pdf_text",
                "find_like", "open_document", "sheet_index"} & took


@needs_sheet_addresses
def test_planlens_own_host_specs_are_the_cores_with_sheet_numbers(sheet_set):
    """planlens' host layer specifies a standing tool that takes sheet
    numbers the way the core shows it: its schema whole. (The two sheet
    checks' ``status`` is shown differently by the two since before this
    work, so they are compared argument by argument in the test above.)"""
    from planlens.review import specs as host_specs
    kit = _kit({SET: sheet_set.pdf}, sheet_tools=True)
    theirs = {s["name"]: s for s in host_specs.host_specs(
        look=True, sheet_tools=True)}
    standing = [n for n in TAKE_SHEETS if not n.startswith("check_")]
    assert len(standing) == 6
    for name in standing:
        assert theirs[name]["input_schema"] == \
            _shown(kit[name])["input_schema"], name


@needs_sheet_addresses
def test_a_sheet_number_finds_its_page_and_a_wrong_one_is_refused(sheet_set):
    kit = _kit({SET: sheet_set.pdf}, sheet_tools=True)
    number = sheet_set.numbers[1]
    read = _call(kit, "read_document", handle=SET, sheets=number)
    assert read["sheet_pages"] == [{"sheet": number, "page": 1,
                                    "pdf_page": 2}]
    assert read["pages_returned"] == "1"
    # the same pages as the page number gives
    by_page = _call(kit, "read_document", handle=SET, pages="1")
    assert read["text"] == by_page["text"]
    # several, as text or as a list
    both = sheet_set.numbers[1:3]
    for given in (", ".join(both), list(both)):
        rows = _call(kit, "document_page_map", handle=SET, sheets=given)
        assert [r["page"] for r in rows["rows"]] == [1, 2]
        assert [r["sheet"] for r in rows["sheet_pages"]] == both
    # a number the index does not hold is refused, and nothing is guessed
    refused = _call(kit, "read_document", handle=SET, sheets="ZZ-999")
    assert "ZZ-999 is not in this document's sheet index" in refused["error"]
    assert "sheet_index" in refused["hint"]
    near = number.replace("-", "")
    refused = _call(kit, "search_document", handle=SET, pattern="PLAN",
                    sheets=near)
    assert "error" in refused and f"{near} -> {number}" in refused["hint"]
    # pages and sheets that name different pages are refused
    refused = _call(kit, "read_document", handle=SET, pages="0",
                    sheets=number)
    assert "name different pages" in refused["error"]
    # nothing given is nothing handed on
    assert "sheet_pages" not in _call(kit, "read_document", handle=SET,
                                      pages="1", sheets="")


@needs_sheet_addresses
def test_the_page_and_its_helper_take_sheet_numbers_behind_the_switch():
    tools, reader = _page(**{SHEETS_ENV: "1"})
    for name in TAKE_SHEETS:
        assert "sheets" in _schema(tools[name])["properties"], name
    # the helper holds the page's own tool objects
    assert "sheets" in _schema(tools["read_document"])["properties"]
    assert {"read_document", "check_cross_references"} <= set(reader)
    unset, _reader = _page()
    assert "sheets" not in _schema(unset["read_document"])["properties"]


# -- drawing_geometry -----------------------------------------------------------

@needs_drawing
def test_the_drawing_tool_is_apart_from_the_standing_list():
    assert document_tools.DRAWING_TOOL_NAMES == (DRAWING,)
    assert DRAWING not in document_tools.document_tool_names()
    assert DRAWING not in document_tools.sheet_tool_names()
    assert DRAWING not in document_tools.PLAN_TOOL_NAMES
    options = document_tools.host_for(drawing_tools=True).toolkit_options
    assert options == {"drawing_tools": True}


@needs_drawing
def test_on_the_page_and_its_reading_helper_have_the_drawing_tool():
    from aec_document_review.review_agent import REVIEW_REWRITES
    from tests.test_review_page_vocabulary_offline import VOCAB
    tools, reader = _page(**{DRAWING_ENV: "1"})
    unset, unset_reader = _page()
    assert set(tools) - set(unset) == {DRAWING}
    assert set(reader) - set(unset_reader) == {DRAWING}
    # every other tool is as it was
    assert {n: _shown(t) for n, t in tools.items() if n != DRAWING} == \
        _all_shown(unset)
    want = (document_tools.tool_description(DRAWING)
            + vision_kit._PDF_PAGE_NOTE)
    assert not REVIEW_REWRITES.get(DRAWING)
    assert tools[DRAWING].description == want
    assert not VOCAB.search(want), VOCAB.search(want).group(0)


@needs_drawing
def test_the_drawing_tool_is_shown_as_planlens_host_layer_specifies_it(
        sheet_set):
    from planlens.review import specs as host_specs
    for sheet_tools in (False, True):
        kit = _kit({SET: sheet_set.pdf}, drawing_tools=True,
                   sheet_tools=sheet_tools)
        theirs = next(s for s in host_specs.host_specs(
            look=True, sheet_tools=sheet_tools, drawing_tools=True)
            if s["name"] == DRAWING)
        mine = _shown(kit[DRAWING])
        assert theirs["description"] == mine["description"]
        assert theirs["input_schema"] == mine["input_schema"]
    assert _schema(kit[DRAWING])["required"] == ["handle"]


@needs_drawing
def test_the_drawing_tool_reads_a_made_up_dimensioned_sheet(tmp_path):
    from planlens.testing import construct_fixtures as cf
    path, _truth = cf.build_synthetic_dimension_pdf(tmp_path)
    with open(path, "rb") as fh:
        kit = _kit({"dims.pdf": fh.read()}, drawing_tools=True)
    found = _call(kit, DRAWING, handle="dims.pdf", pdf_page=1)
    assert found["page"] == 0 and found["pdf_page"] == 1
    assert found["dimensions"] and "measured_from" in found
    assert found["counts"]["dimensions"] == len(found["dimensions"])
    # what was given is handed on, and no more
    only = _call(kit, DRAWING, handle="dims.pdf", page=0,
                 find=["title_block"])
    assert "dimensions" not in only or not only["dimensions"]
    # another host of the same conversation is still not served it
    plain = _kit({"dims.pdf": b""})
    assert DRAWING not in plain


@needs_drawing
@needs_sheet_addresses
def test_the_drawing_tool_takes_a_sheet_number_with_the_sheet_tools(
        sheet_set):
    kit = _kit({SET: sheet_set.pdf}, drawing_tools=True, sheet_tools=True)
    number = sheet_set.numbers[1]
    found = _call(kit, DRAWING, handle=SET, sheet=number)
    assert found["page"] == 1
    assert found["sheet_pages"][0]["sheet"] == number


# -- a DXF also as a plot --------------------------------------------------------

def test_the_host_asks_for_plots_only_a_planlens_that_has_them():
    options = document_tools.host_for(dxf_plots=True).toolkit_options or {}
    assert options.get("dxf_plots", False) is HAVE_DXF_PLOTS


@needs_dxf_plots
def test_on_the_page_says_a_dxf_is_also_a_plot_and_nothing_else_moves():
    from tests.test_review_page_vocabulary_offline import VOCAB
    tools, reader = _page(**{DXF_ENV: "1"})
    unset, unset_reader = _page()
    assert reader == unset_reader and set(tools) == set(unset)
    changed = [n for n in tools if _shown(tools[n]) != _shown(unset[n])]
    assert changed == ["open_document"]
    old, new = vision_kit._DXF_PLOT_REWRITE
    # the phrase the switch replaces is still in the page's description
    assert old in unset["open_document"].description
    assert tools["open_document"].description == \
        unset["open_document"].description.replace(old, new)
    assert _schema(tools["open_document"])["properties"] == \
        _schema(unset["open_document"])["properties"]
    assert not VOCAB.search(new), VOCAB.search(new).group(0)


@needs_dxf_plots
def test_on_the_page_tools_and_the_looks_take_a_dxf_as_its_plot(
        dxf, tmp_path, monkeypatch):
    from planlens.dxf.plot import PLOT_NOTE
    monkeypatch.chdir(tmp_path)          # the marked copy is written here
    att = {DXF: dxf}
    before = _call(_kit(att), "open_document", source=DXF)
    from planlens.review import conversation
    conversation._SPACES.clear()
    kit = _kit(att, dxf_plots=True)
    opened = _call(kit, "open_document", source=DXF)
    # asking adds the plot's line, and the note's last sentence points at
    # the plot instead of saying there are no pages (the review of B15);
    # nothing else changes
    assert opened.pop("plot").endswith(PLOT_NOTE)
    assert opened.pop("note") != before.pop("note")
    assert opened == before
    rows = _call(kit, "document_page_map", handle=DXF)
    assert [r["page"] for r in rows["rows"]] == [0]
    # reading still reads the DXF's own text
    assert "GRID B" in _call(kit, "read_document", handle=DXF)["text"]
    # the two page looks see the plot, and say what it is
    for name, args in (("analyze_pdf_page", {}),
                       ("render_region", {"bbox": [0, 0, 100, 100]})):
        look = _call(kit, name, attachment_key=DXF, page=0,
                     prompt="what is drawn", **args)
        assert look["analysis"] == "seen", look
        assert look["plot"] == PLOT_NOTE
    # the attachments handed in are never written to
    assert att == {DXF: dxf}
    # a marked copy of the plot is a PDF
    marked = _call(kit, "annotate_document", handle=DXF, markups=[
        {"page": 0, "kind": "note", "point": [100, 100],
         "comment": "check this line"}])
    assert marked["n_written"] == 1
    assert marked["output_path"] == "site_marked.pdf"


@needs_dxf_plots
def test_on_a_pdf_look_is_as_it_was_and_a_broken_dxf_is_said(sheet_set):
    att = {SET: sheet_set.pdf, "broken.dxf": b"0\nSECTION\n2\nNONSENSE"}
    plain = _kit(att).get("analyze_pdf_page").invoke(
        {"attachment_key": SET, "page": 1, "prompt": "x"})
    asked = _kit(att, dxf_plots=True)
    assert asked["analyze_pdf_page"].invoke(
        {"attachment_key": SET, "page": 1, "prompt": "x"}) == plain
    refused = _call(asked, "analyze_pdf_page", attachment_key="broken.dxf",
                    page=0, prompt="x")
    assert refused["error"].startswith("could not plot 'broken.dxf'")
    assert "read_document" in refused["hint"]
