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
  drawings and reports (plans to scale, elevation up a profile, kPa and
  kN, units and the coordinate system named; the owner's conventions of
  2026-10-15, BACKLOG GM-97);
- worked EXAMPLES it can fetch and adapt (``figure_examples``, from
  :mod:`profile_figure.figure_examples`): ``plot_data`` calls, and when it
  has ``plot_code``, matplotlib scripts for that box too;
- ``plot_data`` to draw and ``analyze_image`` to LOOK at the PNG, which is
  the same figure the chat shows (both are drawn from one spec), with a
  budget of a few redraws;
- read access to the working folder for data the brief points at.

On the main agent, :class:`FigureTasks` attaches the nearest worked example
to each brief and adds to the helper's result, from its record, what it
drew and what each of its looks found; :data:`FIGURE_PRIMARY_NOTE` tells the
main agent to hand figures to the helper and not to look again at what it
looked at (live wave w9, ``module_work/live_smoke/runs/w9-figures/
TRACE_REVIEW.md`` §2.6, §2.9).

It is ON by default since 2026-10-20 (the owner, GM-92: "on everywhere";
BACKLOG GM-227); ``GEOTECH_FIGURE_HELPER=0`` with ``GEOTECH_PLOT_CODE=0``
leaves it out.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.tools import StructuredTool

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

#: Model calls a figure job may take: brief, example, draw, look, redraw,
#: look, redraw, look, answer, with a little slack.
DEFAULT_FIGURE_MAX_MODEL_CALLS = 14

FIGURE_HELPER_NAME = "figure"

#: What the delegating agent is told (the task tool's list of helpers).
FIGURE_HELPER_DESCRIPTION = (
    "The figure helper. Delegate every figure the user will see or a report "
    "will carry: a plan of borings and a footprint, a profile (SPT, CPT, "
    "su, water content against elevation), results against a parameter, "
    "the cases of a sensitivity study, settlement against time, a load "
    "test, an exceedance curve. Give it the numbers (or the name of a file "
    "in the working folder that holds them), what the figure must show and "
    "for whom, the units, the coordinate system of any plan, and any value "
    "the text will quote (a design value, P90, the governing case). Leave "
    "the styling to it: it applies its own figure checklist (names on the "
    "data rather than a legend, the project's own words, readings joined "
    "only as the data allow), and nothing the brief does not give goes on "
    "the figure, so give no test standard, classification or citation the "
    "user did not supply. It draws with plot_data, LOOKS at the result "
    "(what it shows against the data, then the lettering) and redraws at "
    "most twice, then returns the PNG and chart file names and one line on "
    "what the figure shows; the result also lists, from the record, what "
    "each of its looks found and where the code of each figure is kept. "
    "To change a figure drawn earlier, send it a brief that names the "
    "figure's file and keeps that name: it is handed the code that drew "
    "the figure and redraws it in place. It never makes up data.")

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
   A profile has ELEVATION on its vertical axis, rising upward: borings
   start at different ground levels, so their depths do not line up.
   Depth (downward, depth_axis) only when no elevations are given, and the
   figure says so; never depth and elevation on two axes of one profile.
   Settlement against time increases downward (depth_axis). A wall or pile
   is drawn to scale with its diagrams beside it; the diagram alone only
   when the structure is not known (the structural engineer designs it),
   and the figure says so. An axis that compares magnitudes starts at zero
   unless the variation is the point, and then says so. A log scale only
   for a quantity spanning decades (time, cycles, stress).
5. Name everything a reader needs. Every axis has its quantity and unit:
   stresses and pressures in kPa, forces in kN (kN/m along a wall); a
   plan names its coordinate system (local grid, state plane, elevation
   datum): a local grid is x and y, never easting and northing (those name
   a projected system); the title says what and where in plain words.
6. Lines mean continuity. Separate cases, scenarios or borings compared
   side by side are markers on a case axis (x as names), never joined by a
   line; a line joins values along ONE continuous thing (one boring's
   profile, one test, one curve). Separate tests are points; a line
   through them is a profile someone assumed, named as one (preconsolidation
   stress: test points when plotting the data, a line for an assumed
   design profile; the figure says which). Readings (a value at each date
   or depth) are joined by straight lines or shown as points, never by a
   stair or step unless the data is a step.
7. Small multiples. Several borings, cases or sections compared: the same
   axes, the same scale and limits, the same colours for the same thing.
8. Colour and shape with meaning. One colour per group, the same group the
   same colour in every figure of a job; groups told apart by shape too, so
   the figure survives black-and-white printing.
9. Mark what the text quotes. The design value, P50 / P90, the governing
   case, a capacity: a labelled point or reference line on the figure.
10. The project's own words. Name every group, line and zone as the logs,
   the report and the references name it (top of weathered rock, refusal,
   SPT N60); never coin a label for the figure. Nothing goes on the figure
   that the brief or the files do not give: no test standard,
   classification, boundary or citation added from memory; a line or zone
   from a standard is drawn from the values the brief gives and named as
   the brief names its source.
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
    "in a value; if something the figure needs is missing, say so. The "
    "brief supplies the data and the purpose; the checklist governs the "
    "styling: where the brief asks for styling the checklist rules out (a "
    "legend for named lines, a stair through readings, a label in words "
    "of its own), follow the checklist and say so in your reply.\n"
    "b. Call figure_examples to see the worked examples, take the nearest "
    "one (figure_examples(name=...)) and adapt its plot_data call: it shows "
    "the checklist applied. When the brief ends with a worked example "
    "attached, start from that one.\n"
    "c. Draw with plot_data. Give output_path a short descriptive file name.\n"
    "d. LOOK: analyze_image on the PNG it saved, in a step AFTER the draw, "
    "never beside it. Ask first what the figure SHOWS, against the data you "
    "drew it from: have the look read two or three plotted values and the "
    "axis ranges and units back off the figure, and compare them here with "
    "the data; is anything drawn that the data do not give (a stair or a "
    "smoothed curve between readings, an extrapolation, a line, zone or "
    "label not in the brief or the files)? are the names the brief's own "
    "words? does a legend repeat names already on the data? is every "
    "series and point the brief named there, and the quoted value marked? "
    "Then the lettering: do labels overlap each other or the data, are the "
    "axis titles and units there, is a plan to scale? plot_data draws its "
    "PNG and the chat's interactive chart from one spec, so this checks "
    "what the user sees.\n"
    "e. Fix what the look found and redraw; look again. At most two "
    "redraws, three draws in all whatever the tool or file name: then hand "
    "back the best one and name what you could not fix (and why: a tool "
    "limit, missing data). Redraw under the same name; a draft drawn under "
    "another name is retired with replaces=[its file name] on the next "
    "draw, so the chat shows only the final figure.\n"
    "f. Use calculate for any number you derive (a percentile, a fraction, "
    "a unit conversion) and for counting or averaging a list; never by "
    "eye.\n\n"
    "REPLY with: the PNG file name and the chart file name plot_data "
    "returned; one line on what the figure shows; any checklist item not "
    "met and why. Do not paste the data back."
)


#: How the list names each kind of example.
_KINDS = {"plot_data": "plot_data call", "plot_code": "plot_code script"}


def _example_names(with_code: bool = False) -> list:
    """The examples offered: the plot_data calls, then the plot_code
    scripts when the helper has plot_code."""
    from profile_figure.figure_examples import (code_example_names,
                                                example_names)
    return example_names() + (code_example_names() if with_code else [])


def _closest_example(want: str, with_code: bool = False) -> Optional[str]:
    """The example ``want`` names among those offered: exact, then by
    shared words with the example's name and what it shows, then by
    spelling."""
    import difflib
    import re
    from profile_figure.figure_examples import CODE_EXAMPLES, EXAMPLES
    names = _example_names(with_code)
    shows = {n: (EXAMPLES.get(n) or CODE_EXAMPLES[n])["shows"]
             for n in names}
    key = want.lower().strip().replace(" ", "_").replace("-", "_")
    if key in names:
        return key
    words = set(re.findall(r"[a-z]+", want.lower())) - {"vs", "of", "the",
                                                         "a", "and", "plot"}
    if words:
        def score(n):
            text = set(re.findall(r"[a-z]+", (n + " " + shows[n]).lower()))
            return len(words & text)
        best = max(names, key=score)
        if score(best) > 0:
            return best
    close = difflib.get_close_matches(key, names, n=1, cutoff=0.5)
    return close[0] if close else None


def make_figure_examples_tool(plot_code: bool = False) -> StructuredTool:
    """``figure_examples``: the list of worked examples, or one in full.
    With ``plot_code`` (the helper has that tool) the list also holds the
    worked plot_code scripts."""
    from funhouse_agent.deep.strict_args import strict_tool

    def figure_examples(name: str = "") -> str:
        """List the worked figure examples, or return one in full."""
        from profile_figure.figure_examples import example
        names = _example_names(plot_code)
        if not name:
            listing = []
            for n in names:
                ex = example(n)
                listing.append({"name": n, "kind": _KINDS[ex["kind"]],
                                "shows": ex["shows"]})
            return json.dumps({
                "examples": listing,
                "next": ("figure_examples(name=...) returns one: what it "
                         "shows, why it is drawn that way, and its complete "
                         "plot_data parameters"
                         + (", or its complete plot_code script and the "
                            "data it reads," if plot_code else "")
                         + " to adapt.")}, ensure_ascii=False)
        want = str(name).strip()
        match = _closest_example(want, plot_code)
        if match is None:
            return json.dumps({"error": f"no example '{want}'; there are "
                                        f"{names}"})
        ex = example(match)
        if match != want:
            # live wave w7 (2026-10-15): the helper guessed 'plan',
            # 'settlement_vs_time' and 'sensitivity cases' first
            ex["matched"] = f"'{want}' read as '{match}'"
        return json.dumps(ex, ensure_ascii=False)

    description = (
        "Worked examples of good engineering figures, each a complete "
        "plot_data call on made-up data with why it is drawn that way: "
        "a plan of borings and a footprint, an SPT profile against "
        "elevation, sensitivity cases, settlement against time, a load "
        "test, an exceedance curve. No name: the list. name=<example>: "
        "that one in full. Adapt the nearest one; never copy its made-up "
        "numbers.")
    if plot_code:
        description += (
            " The list also holds complete plot_code scripts (matplotlib, "
            "with the data each reads) for figures plot_data cannot draw; "
            "each entry says which kind it is.")
    return strict_tool(StructuredTool.from_function(
        figure_examples, name="figure_examples", description=description))


#: Added to the helper's way of working when ``plot_code`` is offered.
PLOT_CODE_STEP = (
    "\ng. When plot_data cannot draw what the brief needs, write the figure "
    "as matplotlib code with plot_code (its description says what the "
    "box allows; figure_examples lists worked plot_code scripts to start "
    "from) and apply the same checklist: look at the PNG it saved (its "
    "look_file when it has one), as in d, and redraw within the same cap; "
    "a plot_data draft under another name goes with replaces=. Save a "
    "Plotly chart too only when the brief asks for an "
    "interactive chart: the chat shows it in place of the PNG, and your "
    "looks do not check it.")

#: Added to the helper's description (what the delegating agent reads) when
#: ``plot_code`` is offered: live wave w8 (F55) drew a wall and its pressure
#: diagrams with plot_data five times and ran out of steps, never knowing a
#: helper could draw it in code.
PLOT_CODE_DESCRIPTION_NOTE = (
    " It also draws in code (plot_code) what plot_data cannot: a wall or "
    "pile drawn to scale with its pressure or load diagrams beside it, "
    "panels sharing one elevation axis, a section with its layers filled, "
    "contours.")


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
    from funhouse_agent.deep.limits import DEFAULT_MAX_RESULT_CHARS
    from funhouse_agent.deep.calculate_tool import make_calculate_tool
    from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
    from funhouse_agent.deep.vision_kit import make_vision_tools

    tools = make_vision_tools(
        engine=engine, attachments=attachments, save_fn=save_fn,
        include={"plot_data", "analyze_image", "list_files",
                 "read_text_file"},
        max_result_chars=max_result_chars or DEFAULT_MAX_RESULT_CHARS,
        reference_result_chars=reference_result_chars,
        plot_data_replaces=True,
    )
    if not any(t.name == "plot_data" for t in tools):
        return None
    plot_code = _plot_code_tool()
    tools = list(tools) + [
        make_figure_examples_tool(plot_code=plot_code is not None),
        make_calculate_tool()]
    prompt = FIGURE_HELPER_PROMPT
    description = FIGURE_HELPER_DESCRIPTION
    if plot_code is not None:
        tools.append(plot_code)
        prompt = prompt + PLOT_CODE_STEP
        description = description + PLOT_CODE_DESCRIPTION_NOTE
    if extra_tools:
        tools = tools + list(extra_tools)
    if extra_system_prompt:
        prompt = prompt + "\n\n" + extra_system_prompt
    spec = {"name": FIGURE_HELPER_NAME,
            "description": description,
            "system_prompt": prompt,
            "tools": tools}
    if max_model_calls:
        spec["middleware"] = [ModelCallBudgetMiddleware(max_model_calls)]
    return spec


# ---------------------------------------------------------------------------
# On the main agent: the worked example goes with the brief, the looks come
# back with the result
# ---------------------------------------------------------------------------

#: What the main agent is told when the figure helper is on (live wave w9:
#: the main agent drew drafts itself, prescribed styling the checklist rules
#: out, put a standard nobody gave into a figure, and looked again at
#: figures the helper had looked at, twice telling the user something false).
FIGURE_PRIMARY_NOTE = (
    "FIGURES: a `figure` helper is available (task with subagent_type "
    "'figure'). Hand it every figure the user will see or a report will "
    "carry, rather than drawing a draft yourself first. Give it the data "
    "(or the file that holds them), what the figure must show and for whom, "
    "the units, and any value the text will quote. Leave the styling to its "
    "checklist (do not prescribe a legend, a step line or labels of your "
    "own), and put nothing in the brief that the user did not give: a test "
    "standard, classification or citation the figure needs comes from the "
    "references, cited as found there, or stays off the figure. Its result "
    "lists, from the record, what it drew and what each of its looks found: "
    "do not look again at a figure it has looked at. Look again only to "
    "judge an item its reply leaves open, and when your look disagrees with "
    "its looks, ask the helper (it has the code) rather than tell the user.")

#: Words that say nothing about which figure a brief wants.
_STOP_WORDS = frozenset(
    "a an and are as at be by can each for from has have in into is it its "
    "may not of on one or so than that the their them then there these "
    "this those to two use used using vs versus was were what when where "
    "which while with without your you plot plots figure figures draw "
    "drawn drawing show shows shown report data value values line lines "
    "name named label labels axis axes result results".split())


def _words(text) -> set:
    """The telling words of ``text``: lower case, no stop words, a plural
    read as its singular."""
    out = set()
    for w in re.findall(r"[a-z][a-z0-9]+", str(text or "").lower()):
        if w in _STOP_WORDS:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def nearest_example(brief: str, with_code: bool = False) -> Optional[str]:
    """The worked example nearest a brief (live wave w9: the helper called
    figure_examples in none of five flows). An example is a candidate when
    the brief has every word of its name as a whole word (``time_settlement``:
    time and settlement) and, for a name of ONE word (a word as common in a
    brief as ``cases``), at least two more words of what it shows; the
    candidate sharing the most telling words wins, a word of its name or
    what it shows counting twice one of its reasons. ``None`` when there is
    no candidate: a figure no example is like gets none."""
    from profile_figure.figure_examples import CODE_EXAMPLES, EXAMPLES
    want = _words(brief)
    best, best_score = None, 0
    for name in _example_names(with_code):
        ex = EXAMPLES.get(name) or CODE_EXAMPLES[name]
        named = _words(name.replace("_", " "))
        shown = len(want & (_words(ex["shows"]) - named))
        if not named or not named <= want:
            continue
        if len(named) < 2 and shown < 2:
            continue
        head = len(named) + shown
        score = 2 * head + len(want & _words(" ".join(ex["why"])))
        if score > best_score:
            best, best_score = name, score
    return best


#: How an attached example opens; also how a brief that has one is known.
EXAMPLE_HEADING = "WORKED EXAMPLE ATTACHED"


def brief_with_example(brief: str, with_code: bool = False) -> str:
    """``brief`` with the nearest worked example's reasons after it (what it
    shows and why it is drawn that way, not its made-up numbers); the brief
    unchanged when no example is near or one is attached already."""
    if not isinstance(brief, str) or EXAMPLE_HEADING in brief:
        return brief
    name = nearest_example(brief, with_code)
    if name is None:
        return brief
    from profile_figure.figure_examples import example
    ex = example(name)
    lines = [brief.rstrip(), "",
             f"{EXAMPLE_HEADING} (the nearest to this brief, picked in "
             f"code): '{name}', a {_KINDS[ex['kind']]}. It shows: "
             f"{ex['shows']} Why it is drawn that way:"]
    lines += [f"- {why}" for why in ex["why"]]
    lines.append(f"figure_examples(name='{name}') returns it in full. "
                 "Adapt how it is drawn, never its made-up numbers; the "
                 "brief says what to draw.")
    return "\n".join(lines)


#: The longest part of one look's answer the record repeats.
LOOK_EXCERPT_CHARS = 320
#: The most looks the record lists (the latest).
MAX_LOOKS_LISTED = 8


def _json_of(text):
    """The JSON object a tool result holds; also with a note after it (the
    calc helper's :class:`FigureLooks`) or a key map before it
    (``result_map.LargeResultKeyMap``)."""
    try:
        out = json.loads(text)
    except (TypeError, ValueError):
        out = None
        if isinstance(text, str):
            decoder = json.JSONDecoder()
            for m in re.finditer(r"(?m)^\{", text):
                try:
                    out = decoder.raw_decode(text, m.start())[0]
                    break
                except ValueError:
                    continue
    return out if isinstance(out, dict) else None


def _base(name) -> str:
    return str(name or "").replace("\\", "/").rsplit("/", 1)[-1]


#: Tools besides plot_code and plot_data whose results can name figures a
#: module method drew (the calc helper's profile_figure and calc_package
#: calls).
MODULE_TOOL = "call_agent"

_PICTURES = (".png", ".jpg", ".jpeg")


def _picture_paths(obj, depth: int = 0) -> list:
    """Every ``output_path`` naming a picture, anywhere in a result."""
    out = []
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "output_path" and isinstance(v, str) \
                    and v.lower().endswith(_PICTURES):
                out.append(v)
            elif isinstance(v, (dict, list)):
                out += _picture_paths(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:200]:
            out += _picture_paths(v, depth + 1)
    return out


def _drawn_files(tool: str, text: str):
    """``(files, look copies, code files)`` a successful draw names: the
    figure files by their base names, each look copy's base name -> its
    figure, and each figure -> where its code is kept."""
    res = _json_of(text) or {}
    files, looks, code = [], {}, {}
    if tool == "plot_code":
        for entry in res.get("images") or []:
            if isinstance(entry, dict) and entry.get("file"):
                files.append(_base(entry["file"]))
                if entry.get("look_file"):
                    looks[_base(entry["look_file"])] = _base(entry["file"])
                if entry.get("code_file"):
                    code[_base(entry["file"])] = entry["code_file"]
    elif tool == "plot_data":
        if res.get("output_path"):
            files.append(_base(res["output_path"]))
            if res.get("code_file"):
                code[_base(res["output_path"])] = res["code_file"]
    elif not res.get("error"):
        files = list(dict.fromkeys(_base(p) for p in _picture_paths(res)))
    return files, looks, code


def _code_still_kept(path: str) -> bool:
    """Whether a kept code file is still there (a retired figure's goes)."""
    try:
        from funhouse_agent._fileio import host_output_dir
        root = host_output_dir()
    except Exception:                                      # noqa: BLE001
        root = None
    if not root:
        return True
    import os
    return os.path.isfile(os.path.join(root, path))


def _look_answer(text: str) -> str:
    res = _json_of(text)
    if res is None:
        found = str(text or "")
    elif res.get("error"):
        found = f"(the look failed: {res['error']})"
    else:
        found = res.get("analysis")
        if found is None:
            found = text
        elif not isinstance(found, str):
            found = json.dumps(found, ensure_ascii=False, default=str)
    found = " ".join(str(found).split())
    if len(found) > LOOK_EXCERPT_CHARS:
        found = found[:LOOK_EXCERPT_CHARS].rstrip() + " ..."
    return found


def _record_draws(rec):
    """``(draws, looks, look copies, code files, draws by tool)`` from a
    helper's record: each successful draw's figure files, in order; each
    look as ``(figure, draws before it, what it found)``."""
    from funhouse_agent.plot_box import DRAW_TOOLS, drew_figure
    calls = {}
    for step in list(rec.steps):
        if step.get("kind") == "model":
            for c in step.get("calls") or []:
                calls[c.get("id")] = c.get("args") or {}
    draws, looks, look_copy, code, by_tool = [], [], {}, {}, {}
    for step in list(rec.steps):
        if step.get("kind") != "result":
            continue
        name, text = step.get("name"), step.get("text") or ""
        if name in DRAW_TOOLS or name == MODULE_TOOL:
            if step.get("status") == "error" or (
                    name in DRAW_TOOLS and not drew_figure(text)):
                continue
            files, copies, kept = _drawn_files(name, text)
            if not files and name == MODULE_TOOL:
                continue                     # a method that drew nothing
            look_copy.update(copies)
            code.update(kept)
            draws.append(files)
            by_tool[name] = by_tool.get(name, 0) + 1
        elif name == "analyze_image":
            args = calls.get(step.get("id")) or {}
            target = _base(args.get("attachment_key"))
            figure = look_copy.get(target, target)
            looks.append((figure, len(draws), _look_answer(text)))
    return draws, looks, look_copy, code, by_tool


def looked_at(rec) -> set:
    """The figures (base names) a helper's looks have read so far."""
    if rec is None:
        return set()
    return {figure for figure, _n, _found in _record_draws(rec)[1]}


def looks_report(rec) -> str:
    """What a helper drew, where each figure's code is kept and what each of
    its looks found, written in code from its record
    (:class:`~funhouse_agent.deep.keep_work.HelperRecord`): the main agent
    does not see a helper's steps, and in live wave w9 it looked again at
    figures the helper had looked at up to six times; in w12 it sent a
    second helper to redraw a figure with no code to edit (GM-151). Empty
    when the helper neither drew nor looked."""
    if rec is None:
        return ""
    draws, looks, _copies, code, by_tool = _record_draws(rec)
    last_draw, last_look = {}, {}
    for n, files in enumerate(draws, 1):
        for f in files:
            last_draw[f] = n
    for figure, after, _found in looks:
        last_look[figure] = after
    if not draws and not looks:
        return ""
    tools = ", ".join(f"{k} {v}" for k, v in sorted(by_tool.items()))
    who = ("FIGURE HELPER RECORD (written in code from what the helper "
           "did): it" if rec.subagent == FIGURE_HELPER_NAME else
           f"FIGURE RECORD of the {rec.subagent} helper (written in code "
           "from what it did): it")
    lines = [f"{who} drew {len(draws)} time(s)"
             + (f" ({tools})" if tools else "")
             + f" and looked {len(looks)} time(s)."]
    shown = looks[-MAX_LOOKS_LISTED:]
    first = len(looks) - len(shown)
    for k, (figure, after, found) in enumerate(shown, first + 1):
        lines.append(f"- look {k} at {figure or '(no file named)'}"
                     + (f", after draw {after}" if after else "")
                     + f": {found}")
    unseen = [f for f, n in last_draw.items() if last_look.get(f, -1) < n]
    if unseen:
        lines.append("Not looked at since its last draw: "
                     + ", ".join(unseen) + ".")
    if looks:
        lines.append("Do not look again at what these looks read; look "
                     "only to judge an item the reply leaves open.")
    kept = [(f, c) for f, c in code.items()
            if f in last_draw and _code_still_kept(c)]
    if kept:
        lines.append(
            "The code that drew each figure is kept: "
            + "; ".join(f"{f} in {c}" for f, c in kept[:12])
            + ". To change one, name it in the next helper's brief and keep "
            "its name: that helper is handed this code and redraws it in "
            "place.")
    return "\n".join(lines)


#: Helpers that draw figures: a brief to one of them that names a figure
#: with kept code is handed that code, and its result says what it drew.
DRAWING_HELPERS = (FIGURE_HELPER_NAME, "calc")


def redraw_brief(brief, root=None, named_from=None):
    """``(brief, stems)``: ``brief`` with, for each figure it names (or the
    text ``named_from`` names: the brief as the main agent wrote it) that
    has kept code, that code and what a redraw of it does
    (:func:`funhouse_agent.plot_box.figures.redraw_note`); the brief
    unchanged when it names none or carries the code already."""
    from funhouse_agent.plot_box import figures
    if not isinstance(brief, str) or figures.REDRAW_HEADING in brief:
        return brief, []
    if named_from is None:
        named_from = brief
    if root is None:
        try:
            from funhouse_agent._fileio import host_output_dir
            root = host_output_dir()
        except Exception:                                  # noqa: BLE001
            root = None
    if not root:
        return brief, []
    try:
        stems = figures.named_in(named_from, root)
        note = figures.redraw_note(stems, root) if stems else ""
    except Exception:                                      # noqa: BLE001
        return brief, []
    if not note:
        return brief, []
    return brief.rstrip() + "\n\n" + note, stems


class FigureTasks(AgentMiddleware):
    """On the main agent, when the figure helper is on: each ``task`` for
    the figure helper carries the nearest worked example
    (:func:`brief_with_example`); each ``task`` for a helper that draws
    figures (:data:`DRAWING_HELPERS`) that names a figure with kept code
    carries that code (:func:`redraw_brief`; live wave w12, GM-151); its
    result ends with :func:`looks_report`. Put it just inside
    :class:`~funhouse_agent.deep.keep_work.KeepWork`, whose record of the
    helper it reads."""

    def __init__(self, with_code: bool = False):
        super().__init__()
        self.with_code = bool(with_code)

    @staticmethod
    def _helper(request) -> Optional[str]:
        call = getattr(request, "tool_call", None) or {}
        args = call.get("args")
        if (call.get("name") == "task" and isinstance(args, dict)
                and args.get("subagent_type") in DRAWING_HELPERS):
            return args["subagent_type"]
        return None

    def _briefed(self, request, helper: str):
        call = request.tool_call
        args = call["args"]
        given = args.get("description")
        brief = given
        if helper == FIGURE_HELPER_NAME:
            brief = brief_with_example(brief, self.with_code)
        brief, _stems = redraw_brief(brief, named_from=given)
        if brief == given:
            return request
        try:
            return request.override(tool_call={
                **call, "args": {**args, "description": brief}})
        except Exception:                                  # noqa: BLE001
            return request

    @staticmethod
    def _reported(result):
        from funhouse_agent.deep.keep_work import _with_note, current_helper
        rec = current_helper()
        if rec is None or rec.subagent not in DRAWING_HELPERS:
            return result
        try:
            note = looks_report(rec)
        except Exception:                                  # noqa: BLE001
            note = ""
        return _with_note(result, note) if note else result

    def wrap_tool_call(self, request, handler):
        helper = self._helper(request)
        if helper is None:
            return handler(request)
        return self._reported(handler(self._briefed(request, helper)))

    async def awrap_tool_call(self, request, handler):
        helper = self._helper(request)
        if helper is None:
            return await handler(request)
        return self._reported(await handler(self._briefed(request, helper)))


__all__ = ["FIGURE_CHECKLIST", "FIGURE_HELPER_DESCRIPTION",
           "FIGURE_HELPER_NAME", "FIGURE_HELPER_PROMPT", "PLOT_CODE_STEP",
           "PLOT_CODE_DESCRIPTION_NOTE", "FIGURE_PRIMARY_NOTE",
           "DRAWING_HELPERS",
           "DEFAULT_FIGURE_MAX_MODEL_CALLS", "build_figure_subagent",
           "make_figure_examples_tool", "nearest_example",
           "brief_with_example", "redraw_brief", "looks_report",
           "looked_at", "FigureTasks"]


#: The calc helper's names, in :mod:`funhouse_agent.deep.calc_figures` since
#: INF-30 step 5 (the calc helper is the geotech page's), from their old path.
_MOVED = ("CALC_REDRAW_NOTE", "FigureLooks", "MODULE_FIGURE_LOOK")


def __getattr__(name):
    if name in _MOVED:
        from funhouse_agent.deep import calc_figures
        return getattr(calc_figures, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
