"""BACKLOG DR-35 (the owner's OWN-77, option (b)): Document Review's own
worked examples for the figure helper (drawings, schedules, quantities) in
place of the core's geotechnical ones, behind
``GEOTECH_REVIEW_FIGURE_EXAMPLES`` (off by default).

Every example really draws: the plot_data calls through the helper's own
strict plot_data tool, the script in the real plotting box. With the switch
unset the page's figure helper is handed exactly what it was before.
"""

import json
import re

import pytest

pytest.importorskip("deepagents")

from aec_agent_core import review_flags  # noqa: E402
from aec_agent_core.charts import figure_examples as core_set  # noqa: E402
from aec_agent_core.deep import figure_helper  # noqa: E402
from aec_agent_core.deep.figure_helper import (  # noqa: E402
    FigureTasks, build_figure_subagent, make_figure_examples_tool,
    nearest_example)

from aec_document_review import figure_examples as dr_set  # noqa: E402

ENV = review_flags.FIGURE_EXAMPLES_ENV


def _text_size(ex_set, with_code: bool) -> dict:
    """What the helper can be handed from a set, in characters: the tool's
    description, its listing, and every example in full."""
    tool = make_figure_examples_tool(plot_code=with_code, examples=ex_set)
    names = figure_helper._example_names(with_code, ex_set)
    full = sum(len(tool.invoke({"name": n})) for n in names)
    return {"description": len(tool.description),
            "listing": len(tool.invoke({"name": ""})), "full": full,
            "count": len(names)}


# ---------------------------------------------------------------------------
# The examples draw
# ---------------------------------------------------------------------------

def test_every_example_draws_cleanly_through_the_real_tool(tmp_path,
                                                           monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    spec = build_figure_subagent(examples=dr_set)
    plot = next(t for t in spec["tools"] if t.name == "plot_data")
    names = list(dr_set.EXAMPLES)
    assert len(names) == 5
    for name in names:
        ex = figure_helper._example_of(name, dr_set)
        params = dict(ex["params"], output_path=f"ex_{name}")
        out = json.loads(plot.invoke(params))
        assert out.get("status") == "success", (name, out)
        assert not out.get("warnings"), (name, out["warnings"])
        assert (tmp_path / f"ex_{name}.png").is_file()
        assert (tmp_path / f"ex_{name}.plotly.json").is_file(), name


def test_the_script_draws_cleanly_in_the_box(tmp_path):
    pytest.importorskip("matplotlib")
    from aec_agent_core import plot_box
    if not plot_box.available():
        pytest.skip("the plotting box cannot run here")
    assert list(dr_set.CODE_EXAMPLES) == ["submittal_timeline"]
    for name in dr_set.CODE_EXAMPLES:
        ex = figure_helper._example_of(name, dr_set)
        assert ex["kind"] == "plot_code"
        assert not plot_box.check_code(ex["code"]), name
        out = plot_box.run_plot_code(ex["code"], data=ex["data"], name=name,
                                     root=str(tmp_path))
        assert out["status"] == "success", (name, out.get("error"))
        assert not out.get("layout_check"), (name, out["layout_check"])
        assert not out.get("refused") and not out.get("warnings"), out
        (image,) = out["images"]
        assert image["file"] == f"{name}.png"
        assert (tmp_path / image["file"]).is_file()


# ---------------------------------------------------------------------------
# The examples follow the checklist
# ---------------------------------------------------------------------------

def test_the_examples_follow_the_figure_checklist():
    for name in list(dr_set.EXAMPLES) + list(dr_set.CODE_EXAMPLES):
        ex = figure_helper._example_of(name, dr_set)
        assert ex["shows"] and len(ex["why"]) >= 3, name
    plan = dr_set.EXAMPLES["plan_dimensions"]["params"]
    # a plan is a drawing, to scale, its coordinate system and north named
    assert plan["plan_view"] is True and plan["north_deg"] is not None
    assert "metres" in plan["xlabel"] and plan["scale_units"] == "m"
    # the distance written on the plan is the one between its two points
    stairs = next(s for s in plan["series"] if s["label"] == "Stair centres")
    (x0, x1), (y0, y1) = stairs["x"], stairs["y"]
    written = next(a["text"] for a in plan["annotations"]
                   if a.get("color"))
    assert written == f"{((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5:.2f} m"
    for name in ("schedule_count", "comments_by_sheet",
                 "strength", "floor_level_by_sheet"):
        p = dr_set.EXAMPLES[name]["params"]
        assert p["title"] and p["xlabel"] and p["ylabel"], name
        # separate things on a case axis are markers, never joined
        for s in p["series"]:
            assert s["style"] == "markers", name
            assert all(isinstance(v, str) for v in s["x"]), name
        # counts and values against a limit start at zero; the one axis
        # that does not says so
        if name == "floor_level_by_sheet":
            assert p["ylim"][0] > 0 and "spans" in p["footnote"]
        else:
            assert p["ylim"][0] == 0, name
    # a limit or a reference is a line, named on it or, where the points
    # sit on it, by a note giving its value
    for name in ("strength", "floor_level_by_sheet"):
        p = dr_set.EXAMPLES[name]["params"]
        (line,) = p["hlines"]
        named = line.get("label") or next(
            a["text"] for a in p["annotations"]
            if a["y"] < line["value"] < a["y"] + 0.05)
        numbers = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", named)]
        assert line["value"] in numbers, (name, named)
    # the comment counts add up to what the footnote says
    comments = dr_set.EXAMPLES["comments_by_sheet"]["params"]
    total = sum(sum(s["y"]) for s in comments["series"])
    assert f"{total} comments" in comments["footnote"]


def test_a_brief_finds_its_example():
    briefs = {
        "Redraw the Level 1 plan to scale from the grid dimensions and "
        "show the distance between the stairs": "plan_dimensions",
        "Plot the door schedule against the count of door tags on the "
        "floor plans": "schedule_count",
        "Show the review comments on each sheet": "comments_by_sheet",
        "The floor level each sheet gives for Level 2":
            "floor_level_by_sheet",
        "Check the floor levels agree from sheet to sheet":
            "floor_level_by_sheet",
        "Plot the concrete strengths against the specified strength":
            "strength",
        "Compare the cylinder breaks with the specified compressive "
        "strength": "strength",
        "Plot the concrete strengths against the minimum the "
        "specification requires": "strength",
    }
    for brief, name in briefs.items():
        assert nearest_example(brief, examples=dr_set) == name, brief
    assert nearest_example("A timeline of the submittal reviews",
                           with_code=True,
                           examples=dr_set) == "submittal_timeline"


def test_briefs_of_another_kind_get_no_example_of_this_set():
    """A brief no example here is like gets none (DR-35 review): a sheet's
    revision level is not a floor level, and figures of a geotechnical
    report sent to this page with the switch on find nothing to adapt."""
    for brief in (
            "Show the revision level of each sheet",
            "Plot the unconfined compressive strength of the rock cores "
            "against depth",
            "Unconfined compressive strength test results against depth",
            "Undrained shear strength profile against elevation for B-1 "
            "to B-3",
            "Plot the shear strength from the direct shear tests against "
            "normal stress",
            "Plot the strength gain of the mix over 28 days",
            "Factor of safety for the five cases against the required 1.5",
            "Settlement against time for the three plates",
            "SPT N60 against elevation for B-1 to B-3",
            "Bearing capacity against footing width, with the allowable "
            "bearing pressure marked"):
        for with_code in (False, True):
            assert nearest_example(brief, with_code=with_code,
                                   examples=dr_set) is None, brief


def test_the_text_stays_near_the_core_sets_size(capsys):
    """The point is fit, not more tokens: what the helper can be handed
    from this set is no larger than what it is handed today."""
    sizes = {}
    for with_code in (False, True):
        core = _text_size(core_set, with_code)
        mine = _text_size(dr_set, with_code)
        sizes[with_code] = (core, mine)
        assert mine["listing"] <= core["listing"], (core, mine)
        assert mine["full"] <= core["full"], (core, mine)
        assert mine["description"] <= core["description"] * 1.1
    with capsys.disabled():
        for with_code, (core, mine) in sizes.items():
            print(f"\nDR-35 sizes (plot_code={with_code}): core {core}; "
                  f"Document Review {mine}")


# ---------------------------------------------------------------------------
# The page: switch unset, as before; switch on, its own
# ---------------------------------------------------------------------------

def _figure_parts(settings):
    from tests.page_builds import build
    _agent, calls, _dir = build("review_default", capture=True,
                                settings=settings)
    for call in calls:
        kw = call["kwargs"]
        for s in kw.get("subagents") or []:
            if isinstance(s, dict) and s.get("name") == "figure":
                tasks = [m for m in kw.get("middleware") or []
                         if isinstance(m, FigureTasks)]
                return s, tasks
    raise AssertionError("no figure helper in the build")


def test_unset_the_page_is_handed_what_it_was():
    spec, (tasks,) = _figure_parts({})
    tool = next(t for t in spec["tools"] if t.name == "figure_examples")
    with_code = any(t.name == "plot_code" for t in spec["tools"])
    before = make_figure_examples_tool(plot_code=with_code)
    assert tool.description == before.description
    assert tool.invoke({"name": ""}) == before.invoke({"name": ""})
    assert tasks.examples is None


def test_switched_on_the_page_gets_its_own_examples():
    off, _ = _figure_parts({})
    spec, (tasks,) = _figure_parts({ENV: "1"})
    tool = next(t for t in spec["tools"] if t.name == "figure_examples")
    with_code = any(t.name == "plot_code" for t in spec["tools"])
    listed = [e["name"] for e in json.loads(
        tool.invoke({"name": ""}))["examples"]]
    assert listed == list(dr_set.EXAMPLES) + (
        list(dr_set.CODE_EXAMPLES) if with_code else [])
    assert dr_set.TOOL_KINDS in tool.description
    assert "borings" not in tool.description
    assert tasks.examples is dr_set
    # only the examples change: the helper's prompt and description do not
    assert spec["system_prompt"] == off["system_prompt"]
    assert spec["description"] == off["description"]
