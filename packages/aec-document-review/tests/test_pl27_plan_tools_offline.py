"""planlens' plan tools on the Document Review page
(``GEOTECH_REVIEW_PLAN_TOOLS``, BACKLOG PL-27), offline.

Behind the switch the page (and its reading helper) gets planlens'
``plan_symbols`` (every small symbol on a vector plan sheet with its exact
centre and its label) and ``plan_grid`` (page points turned into local
easting / northing, right-handed). Off, or on a planlens without them, the
page is the default build. A planlens after 0.16.0 serves them only to a
host that asks, and the page asks it; 0.16.0 serves them to every host and
is not asked (its toolkit would refuse the option). The tests run the
tools on planlens' own made-up plan sheet, where every symbol is known.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from aec_agent_core import document_tools, review_flags  # noqa: E402
from aec_agent_core.deep import vision_kit  # noqa: E402
from aec_document_review.review_agent import REVIEW_REWRITES  # noqa: E402
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _model, _review_kwargs, _tools_of)

PLAN = ("plan_symbols", "plan_grid")
ENV = "GEOTECH_REVIEW_PLAN_TOOLS"
NAME = "site_plan_sheet_c1.pdf"

#: The installed planlens publishes the plan tools (every planlens the core
#: takes does; kept as a guard).
HAVE = document_tools.plan_tools_available()
needs_plan_tools = pytest.mark.skipif(
    not HAVE, reason="the installed planlens has no plan tools")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.LATER_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert review_flags.plan_tools()


@pytest.fixture(scope="module")
def plan():
    from planlens.testing.plan_location_fixtures import (
        build_plan_location_sheet)
    return build_plan_location_sheet()


def _agent(plan):
    return build_review_page_agent(
        _model([]), attachments={NAME: plan.pdf}, engine=FakeEngine(),
        **_review_kwargs())


def _kit(plan, **kw):
    return {t.name: t for t in vision_kit.make_vision_tools(
        engine=FakeEngine(), attachments={NAME: plan.pdf}, **kw)}


def _schema(tool):
    schema = tool.tool_call_schema
    if not isinstance(schema, dict):
        schema = schema.model_json_schema()
    return schema


def _shown(tool):
    import inspect
    return {"description": inspect.cleandoc(tool.description),
            "input_schema": _schema(tool)}


# -- the switch -----------------------------------------------------------------

def test_the_switch_is_a_later_one_and_off_unless_set(monkeypatch):
    assert review_flags.PLAN_TOOLS_ENV == ENV
    assert ENV in review_flags.LATER_ENVS
    # the geotech app's tests pin these two whole
    assert ENV not in review_flags.ALL_ENVS
    assert not any(ENV in arm for arm in review_flags.ARMS.values())
    assert not review_flags.plan_tools()
    assert ENV not in review_flags.describe()
    for off in ("0", "false", "no", ""):
        monkeypatch.setenv(ENV, off)
        assert not review_flags.plan_tools()
    monkeypatch.setenv(ENV, "1")
    assert review_flags.plan_tools()
    assert f"{ENV}=1" in review_flags.describe()
    with review_flags.switches({}):
        assert not review_flags.plan_tools()
    with review_flags.switches({ENV: "1"}):
        assert review_flags.plan_tools()


def test_the_names_are_apart_from_the_standing_list():
    assert document_tools.PLAN_TOOL_NAMES == PLAN
    # planlens publishes them in its standing list, so they are kept out
    # of the core's: nothing that reads it offers them unasked
    assert not set(PLAN) & set(document_tools.OPTIONAL_DOCUMENT_TOOL_NAMES)
    assert not set(PLAN) & set(document_tools.document_tool_names())


# -- off ------------------------------------------------------------------------

def test_off_neither_the_page_nor_the_kit_has_them(plan):
    assert not set(PLAN) & set(_tools_of(_agent(plan)))
    plain = _kit(plan)
    assert not set(PLAN) & set(plain)
    # the default of the new argument is the build as it was
    asked_off = _kit(plan, plan_tools=False)
    assert {n: _shown(t) for n, t in plain.items()} == \
        {n: _shown(t) for n, t in asked_off.items()}
    assert "plan_tools" not in (document_tools.host_for().toolkit_options
                                or {})


def test_on_without_them_in_planlens_the_page_is_the_default_build(
        on, plan, monkeypatch):
    off_names = set(_kit(plan))
    monkeypatch.setattr(document_tools, "plan_tools_available",
                        lambda: False)
    tools = _tools_of(_agent(plan))
    assert not set(PLAN) & set(tools)
    assert "open_document" in tools
    assert set(_kit(plan, plan_tools=True)) == off_names


# -- on -------------------------------------------------------------------------

@needs_plan_tools
def test_on_the_page_and_its_reading_helper_have_them(on):
    from tests import page_builds
    agent, calls, _ = page_builds.build("review_default", capture=True,
                                        settings={ENV: "1"})
    kw = next(c for c in calls if c["kind"] == "create_deep_agent")["kwargs"]
    by_name = {getattr(t, "name", None): t for t in kw["tools"]}
    assert set(PLAN) <= set(by_name)
    # described in planlens' own words, with the viewer's page number, and
    # on this page with its phrase swap
    for name in PLAN:
        want = (document_tools.tool_description(name)
                + vision_kit._PDF_PAGE_NOTE)
        for old, new in REVIEW_REWRITES.get(name, ()):
            assert old in want, (name, old)
            want = want.replace(old, new)
        assert by_name[name].description == want
    reader = next(s for s in kw["subagents"]
                  if s.get("name") == "general-purpose")
    assert set(PLAN) <= {t.name for t in reader["tools"]}


@needs_plan_tools
def test_on_the_page_names_no_geotech_report_things(on, plan):
    from tests.test_review_page_vocabulary_offline import VOCAB
    tools = _tools_of(_agent(plan))
    for name in PLAN:
        text = tools[name].description
        assert not VOCAB.search(text), (name, VOCAB.search(text).group(0))


@needs_plan_tools
def test_on_each_argument_is_shown_as_planlens_shows_it(plan):
    kit = _kit(plan, plan_tools=True)
    for name in PLAN:
        spec = document_tools._spec(name)["parameters"]["properties"]
        props = _schema(kit[name])["properties"]
        assert set(props) == set(spec) | {"pdf_page"}
        for arg in document_tools.PLAN_SHOWN_FROM_PLANLENS[name]:
            got = dict(props[arg])
            assert got.pop("default") is None
            assert got.pop("title") == arg.replace("_", " ").title()
            assert got == spec[arg], (name, arg)
        assert props["pdf_page"]["anyOf"] == [{"type": "integer"},
                                              {"type": "null"}]
    assert _schema(kit["plan_symbols"])["required"] == ["handle"]
    assert not _schema(kit["plan_grid"]).get("required")


@needs_plan_tools
def test_on_planlens_own_host_specs_are_the_cores_word_for_word(plan):
    """A planlens with the plan tools in its host layer (after 0.16.0)
    specifies them as the core shows them, as it does every document
    tool (its ``app_tool_specs.json``)."""
    from planlens.review import specs as host_specs
    if not hasattr(host_specs, "PLAN_TOOLS"):
        pytest.skip("the installed planlens' host layer has no plan tools")
    kit = _kit(plan, plan_tools=True)
    theirs = {s["name"]: s for s in host_specs.host_specs(
        look=True, plan_tools=True) if s["name"] in PLAN}
    for name in PLAN:
        mine = _shown(kit[name])
        assert theirs[name]["description"] == mine["description"]
        assert theirs[name]["input_schema"] == mine["input_schema"], name


@needs_plan_tools
def test_on_they_read_the_plan_with_a_file_name_and_a_viewer_page(plan):
    kit = _kit(plan, plan_tools=True)
    found = json.loads(kit["plan_symbols"].invoke(
        {"handle": NAME, "pdf_page": 1}))
    assert found["page"] == 0 and found["pdf_page"] == 1, found
    assert {t.label for t in plan.symbols} <= {
        s.get("label") for s in found["symbols"]}
    pts = [{"name": s["label"], "page": s["center"]}
           for s in found["symbols"] if s.get("label")]
    grid = json.loads(kit["plan_grid"].invoke(
        {"controls": plan.controls, "points": pts, "handle": NAME,
         "page": 0, "units": "m"}))
    assert grid["handedness"] == "right" and grid["check"]["right_handed"]
    assert "point_flags" not in grid
    by = {p["name"]: p for p in grid["points"]}
    for t in plan.symbols:
        if t.label not in ("B-1", "TP-1", "BH-2"):
            continue
        assert abs(by[t.label]["easting"] - t.easting) < 0.5, t.label
        assert abs(by[t.label]["northing"] - t.northing) < 0.5, t.label
    # an argument not given is not handed on: planlens' defaults apply
    swapped = [{"page": c["page"], "local": c["local"][::-1]}
               for c in plan.controls]
    refused = json.loads(kit["plan_grid"].invoke({"controls": swapped}))
    assert "MIRRORED" in refused["error"]


def test_the_host_asks_only_a_planlens_that_serves_them_on_asking():
    asks = document_tools.plan_tools_need_asking()
    options = document_tools.host_for(plan_tools=True).toolkit_options or {}
    assert options.get("plan_tools", False) is asks
    assert "plan_tools" not in (document_tools.host_for(
        sheet_tools=True).toolkit_options or {})
