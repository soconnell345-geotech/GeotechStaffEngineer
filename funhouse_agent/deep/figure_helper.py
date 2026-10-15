"""The figure helper: draws a figure, looks at it, redraws it to a checklist.

The owner's choice on 2026-10-15 (OWN-61, option B), after a field session
whose plans were not to scale, whose footprint and boring names could not be
drawn, whose sensitivity cases were joined by a line, and whose drawing
helper never looked at what it drew: "Go with B next, the figure helper. I
assume the helper would be the Tufte checklist and some actual good looking
example plots."

So the helper has:

- a FIGURE CHECKLIST in its own prompt, in our own words, naming the
  principles of Tufte's *The Visual Display of Quantitative Information*
  (Graphics Press, 1983; 2nd ed. 2001) — data-ink, chartjunk, small
  multiples, labels on the data — beside the conventions of engineering
  drawings and reports (plans to scale, depth downward, units and the
  coordinate system named);
- worked EXAMPLES it can fetch and adapt (``figure_examples``, from
  :mod:`profile_figure.figure_examples`);
- ``plot_data`` to draw and ``analyze_image`` to LOOK at the PNG, which is
  the same figure the chat shows (both are drawn from one spec), with a
  budget of a few redraws;
- read access to the working folder for data the brief points at.

It is OFF until switched on (``GEOTECH_FIGURE_HELPER=1``; owner rule: new
agent behaviour waits behind a switch until it is measured).
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import StructuredTool

#: Model calls a figure job may take: brief, example, draw, look, redraw,
#: look, redraw, look, answer, with a little slack.
DEFAULT_FIGURE_MAX_MODEL_CALLS = 14

FIGURE_HELPER_NAME = "figure"

#: What the delegating agent is told (the task tool's list of helpers).
FIGURE_HELPER_DESCRIPTION = (
    "The figure helper. Delegate every figure the user will see or a report "
    "will carry: a plan of borings and a footprint, a depth profile (SPT, "
    "CPT, su, water content against depth), results against a parameter, "
    "the cases of a sensitivity study, settlement against time, a load "
    "test, an exceedance curve. Give it the numbers (or the name of a file "
    "in the working folder that holds them), what the figure must show and "
    "for whom, the units, the coordinate system of any plan, and any value "
    "the text will quote (a design value, P90, the governing case). It "
    "draws with plot_data, LOOKS at the result and redraws until it passes "
    "a figure checklist, then returns the PNG and chart file names and one "
    "line on what the figure shows. It never makes up data.")

#: The checklist, in our own words (Tufte's terms named, not his text).
FIGURE_CHECKLIST = """\
FIGURE CHECKLIST (after Tufte, The Visual Display of Quantitative
Information, 1983/2001, and engineering drawing practice). Check every item
against the PNG you drew, not against what you meant to draw.

1. The data first. Everything the brief asks to show is on the figure and
   nothing hides it: no legend, label or heavy grid over the data.
2. Data-ink. Every mark carries data or helps read it. No background fills,
   heavy frames, 3-D, shadows or decorative colour; a light grid only where
   values are read off; no redundant legend for a single series (chartjunk).
3. Labels on the data. Name lines and points on the figure itself
   ('labels', at a point where the line stands apart from the others), not
   in a legend the reader has to match. Keep a legend only for groups (used
   / not used, two interpretations). Reference lines are named on the line.
4. Scales tell the truth. A section is drawn to scale (equal_aspect: one
   metre across = one metre up). A plan of a site, compound or property is
   a DRAWING, not a chart (plan_view): no axes, axis numbers or grid; a
   scale bar; a north arrow at the angle the plan the data came from (or
   the project's other plans) shows (north_deg; straight up only when no
   plan shows north, and then say so); the property line as a dash-dot
   outline; a zone of interest hatched over what lies under it; faint
   grey for context; the coordinate system named under the scale bar
   (xlabel).
   Depth and settlement
   increase downward (depth_axis). An axis that compares magnitudes starts
   at zero unless the variation is the point, and then says so. A log
   scale only for a quantity spanning decades (time, cycles, stress).
5. Name everything a reader needs. Every axis has its quantity and unit; a
   plan names its coordinate system (local grid, state plane, elevation
   datum): a local grid is x and y, never easting and northing (those name
   a projected system); the title says what and where in plain words.
6. Lines mean continuity. Separate cases, scenarios or borings compared
   side by side are markers on a case axis (x as names), never joined by a
   line; a line joins values along ONE continuous thing (one boring's
   profile, one test, one curve).
7. Small multiples. Several borings, cases or sections compared: the same
   axes, the same scale and limits, the same colours for the same thing.
8. Colour and shape with meaning. One colour per group, the same group the
   same colour in every figure of a job; groups told apart by shape too, so
   the figure survives black-and-white printing.
9. Mark what the text quotes. The design value, P50 / P90, the governing
   case, a capacity: a labelled point or reference line on the figure.
10. The project's own words. Name every group, line and zone as the logs,
   the report and the references name it (top of weathered rock, refusal,
   SPT N60); never coin a label for the figure.
11. Readable at report size. No overlapping text; names inside the frame;
   the legend outside a map; hover text in the interactive chart for what
   the figure cannot show (a boring's depth to rock).
"""

FIGURE_HELPER_PROMPT = (
    "You are the FIGURE HELPER of a geotechnical agent. You draw figures that "
    "an engineer would put in a report, from data you are given.\n\n"
    + FIGURE_CHECKLIST +
    "\nHOW TO WORK\n"
    "a. Read the brief: what must the figure show, to whom, which numbers, "
    "which units and coordinate system, which value the text quotes. If the "
    "data are in a file, read it (list_files, read_text_file). Use only "
    "numbers from the brief or the files: never invent, round off or fill "
    "in a value; if something the figure needs is missing, say so.\n"
    "b. Call figure_examples to see the worked examples, take the nearest "
    "one (figure_examples(name=...)) and adapt its plot_data call: it shows "
    "the checklist applied.\n"
    "c. Draw with plot_data. Give output_path a short descriptive file name.\n"
    "d. LOOK: analyze_image on the PNG it saved, asking specific questions "
    "from the checklist (is every series and point the brief named visible? "
    "do any labels overlap each other or the data? are the axis titles and "
    "units there? is the plan to scale? is the quoted value marked?). The "
    "PNG and the chat's interactive chart are drawn from the same spec, so "
    "this checks what the user sees.\n"
    "e. Fix what the look found and redraw; look again. At most two "
    "redraws: then hand back the best one and name what you could not fix "
    "(and why: a tool limit, missing data).\n"
    "f. Use calculate for any number you derive (a percentile, a fraction, "
    "a unit conversion) and for counting or averaging a list; never by "
    "eye.\n\n"
    "REPLY with: the PNG file name and the chart file name plot_data "
    "returned; one line on what the figure shows; any checklist item not "
    "met and why. Do not paste the data back."
)


def _closest_example(want: str) -> Optional[str]:
    """The example ``want`` names: exact, then by shared words with the
    example's name and what it shows, then by spelling."""
    import difflib
    import re
    from profile_figure.figure_examples import EXAMPLES, example_names
    names = example_names()
    key = want.lower().strip().replace(" ", "_").replace("-", "_")
    if key in names:
        return key
    words = set(re.findall(r"[a-z]+", want.lower())) - {"vs", "of", "the",
                                                         "a", "and", "plot"}
    if words:
        def score(n):
            text = set(re.findall(r"[a-z]+", (n + " " + EXAMPLES[n]["shows"])
                                  .lower()))
            return len(words & text)
        best = max(names, key=score)
        if score(best) > 0:
            return best
    close = difflib.get_close_matches(key, names, n=1, cutoff=0.5)
    return close[0] if close else None


def make_figure_examples_tool() -> StructuredTool:
    """``figure_examples``: the list of worked examples, or one in full."""
    from funhouse_agent.deep.strict_args import strict_tool

    def figure_examples(name: str = "") -> str:
        """List the worked figure examples, or return one in full."""
        from profile_figure.figure_examples import (EXAMPLES, example,
                                                    example_names)
        if not name:
            return json.dumps({
                "examples": [{"name": n, "shows": EXAMPLES[n]["shows"]}
                             for n in example_names()],
                "next": "figure_examples(name=...) returns one: what it "
                        "shows, why it is drawn that way, and its complete "
                        "plot_data parameters to adapt."})
        want = str(name).strip()
        match = _closest_example(want)
        if match is None:
            return json.dumps({"error": f"no example '{want}'; there are "
                                        f"{example_names()}"})
        ex = example(match)
        if match != want:
            # live wave w7 (2026-10-15): the helper guessed 'plan',
            # 'settlement_vs_time' and 'sensitivity cases' first
            ex["matched"] = f"'{want}' read as '{match}'"
        return json.dumps(ex, ensure_ascii=False)

    return strict_tool(StructuredTool.from_function(
        figure_examples, name="figure_examples",
        description=(
            "Worked examples of good engineering figures, each a complete "
            "plot_data call on made-up data with why it is drawn that way: "
            "a plan of borings and a footprint, a depth profile, sensitivity "
            "cases, settlement against time, a load test, an exceedance "
            "curve. No name: the list. name=<example>: that one in full. "
            "Adapt the nearest one; never copy its made-up numbers.")))


#: Added to the helper's way of working when ``plot_code`` is offered.
PLOT_CODE_STEP = (
    "\ng. When plot_data cannot draw what the brief needs, write the figure "
    "as matplotlib code with plot_code (its description says what the "
    "box allows) and apply the same checklist: look at the PNG it saved "
    "and redraw.")


def _plot_code_tool():
    """``plot_code`` when its switch is on and the box can run here."""
    from funhouse_agent import review_flags
    if not review_flags.plot_code():
        return None
    from funhouse_agent import plot_box
    if not plot_box.available():
        return None
    return plot_box.make_plot_code_tool()


def build_figure_subagent(
    engine=None,
    attachments=None,
    save_fn=None,
    max_result_chars: Optional[int] = None,
    reference_result_chars: Optional[int] = None,
    max_model_calls: Optional[int] = DEFAULT_FIGURE_MAX_MODEL_CALLS,
    extra_tools=None,
    extra_system_prompt: Optional[str] = None,
) -> Optional[dict]:
    """The figure helper's spec, or None where plot_data cannot be built."""
    from funhouse_agent.deep.agent import DEFAULT_MAX_RESULT_CHARS
    from funhouse_agent.deep.calculate_tool import make_calculate_tool
    from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
    from funhouse_agent.deep.tools import make_vision_tools

    tools = make_vision_tools(
        engine=engine, attachments=attachments, save_fn=save_fn,
        include={"plot_data", "analyze_image", "list_files",
                 "read_text_file"},
        max_result_chars=max_result_chars or DEFAULT_MAX_RESULT_CHARS,
        reference_result_chars=reference_result_chars,
    )
    if not any(t.name == "plot_data" for t in tools):
        return None
    tools = list(tools) + [make_figure_examples_tool(),
                           make_calculate_tool()]
    prompt = FIGURE_HELPER_PROMPT
    plot_code = _plot_code_tool()
    if plot_code is not None:
        tools.append(plot_code)
        prompt = prompt + PLOT_CODE_STEP
    if extra_tools:
        tools = tools + list(extra_tools)
    if extra_system_prompt:
        prompt = prompt + "\n\n" + extra_system_prompt
    spec = {"name": FIGURE_HELPER_NAME,
            "description": FIGURE_HELPER_DESCRIPTION,
            "system_prompt": prompt,
            "tools": tools}
    if max_model_calls:
        spec["middleware"] = [ModelCallBudgetMiddleware(max_model_calls)]
    return spec


__all__ = ["FIGURE_CHECKLIST", "FIGURE_HELPER_DESCRIPTION",
           "FIGURE_HELPER_NAME", "FIGURE_HELPER_PROMPT", "PLOT_CODE_STEP",
           "DEFAULT_FIGURE_MAX_MODEL_CALLS", "build_figure_subagent",
           "make_figure_examples_tool"]
