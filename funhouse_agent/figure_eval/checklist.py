"""What an engineer marks on a figure, asked of a vision model (GM-314).

The items are the figure helper's ``FIGURE_CHECKLIST``
(``aec_agent_core.deep.figure_helper``) and the owner's figure conventions
of 2026-10-15, split into separate questions a reviewer answers yes, no or
not applicable, each with a one-line reason. A request adds its own items
(``extra``: what it says must be on the figure).

:func:`check_run` sends one run's final figures (:func:`choose_figures`:
the PNGs the agent's reply names by file name, else the newest; never the
figure helper's look copies; at most :data:`MAX_FIGURES`) with the
request, the agent's reply and
the numbers GM-328's build of the same inputs returned, and reads the marks
back. A run that drew nothing is not sent: every item is marked ``no``,
"no figure appeared" (:func:`no_figure_marks`). A run that failed (an
error, a rate limit, a timeout, a crash) is marked the same way and flagged
``run_failed``: the comparison keeps it apart. An item the reply leaves out
is ``unmarked``, never ``na``; a reply that marks fewer than half the items
is flagged ``not_scored`` (the runner asks once more).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from funhouse_agent.chart_eval import imaging
from funhouse_agent.chart_eval.judge import _outer_object, _text_of

#: (id, the question). Order is the report's.
ITEMS = (
    ("quantity_axes", "The right quantity is plotted against the right axis "
     "for what was asked (the input varied runs across, the result up; a "
     "profile's depth or elevation runs up the page)."),
    ("axes_units", "Every axis names its quantity and its unit."),
    ("kpa_kn", "Stresses and pressures are in kPa and forces in kN (kN/m "
     "along a wall, kN-m for moments); no psf, ksf, tsf, psi or kips."),
    ("vertical_elevation", "A profile (a quantity against depth or "
     "elevation) has ELEVATION on its vertical axis, rising upward, when "
     "the request gives a ground elevation; depth (downward) only when no "
     "elevation is given, and then the figure says so; never depth and "
     "elevation on two axes of one profile. 'na' when the figure is not a "
     "profile."),
    ("structure_drawn", "The wall, pile, shaft or footing is drawn, to "
     "scale, with its diagrams beside it; a diagram alone only when the "
     "structure is not known, and then the figure says so. 'na' when the "
     "request involves no structure."),
    ("to_scale", "A cross-section, wall or layered section is drawn to "
     "scale (one metre across equals one metre up). 'na' when the figure "
     "is a chart, not a section."),
    ("labels_terms", "Labels are legible at report size (no overlapping "
     "text, names inside the frame) and use the request's and the "
     "references' own terms (layer names as given, standard symbols), no "
     "coined jargon."),
    ("lines_named", "Lines and points are named on the figure itself, "
     "beside the line or point, with a legend only for a group of lines; "
     "a reference line (a limit, a design value, a water level) is named "
     "on the line. 'na' when the figure has a single line or set of "
     "points and no reference line."),
    ("defined_words", "Every plotted quantity is defined in words (an axis "
     "title, a label or a note), not only by a bare symbol."),
    ("data_ink", "No unnecessary ink: no background fill, heavy frame, 3-D, "
     "shadow or decorative colour, no heavy grid, no legend for a single "
     "series."),
    ("plan_drawing", "A plan is drawn as a drawing: no axes, axis numbers, "
     "grid or frame; a scale bar; a north arrow (straight up only when no "
     "plan shows north, and the figure says so); a property line dash-dot "
     "where the plan shows one; "
     "the coordinate system named once. 'na' when the figure is not a "
     "plan."),
    ("cases_markers", "Separate cases, scenarios or tests are markers on a "
     "case axis (or points), never joined by a line as if continuous. 'na' "
     "when the figure compares no separate cases or tests."),
    ("numbers_match", "Numbers written on the figure, or read off it at a "
     "labelled point, agree with the calculation's result given below "
     "(within reading accuracy). 'na' when the figure states no number "
     "that the result covers."),
)

ITEM_IDS = tuple(i for i, _ in ITEMS)
MARKS = ("yes", "no", "na")
#: An item the checker left without a readable mark.
UNMARKED = "unmarked"
MAX_FIGURES = 4
#: Where the figure helper's plot_code box keeps the smaller copy of a
#: figure it looks at (``aec_agent_core.plot_box.child``: ``.scratch/<name>
#: _look.png``): never a figure the reply presents.
LOOK_DIR = ".scratch"
LOOK_SUFFIX = "_look.png"
NO_FIGURE = "no figure appeared"

PROMPT = """You are a senior geotechnical engineer marking figures that an assistant drew for a report, as you would mark them in a review. Judge only what you SEE in the figure image(s), not what the reply says the figure shows.

THE REQUEST
{request}

THE CALCULATION'S RESULT for these inputs (from the calculation tool; use it for the item numbers_match):
{numbers}

THE ASSISTANT'S REPLY (for context only):
{reply}

{n_figures} figure(s) follow, in this order: {names}. Mark the figure(s) together, as the answer to the request.

Mark each item 'yes', 'no' or 'na' (not applicable to this figure), with a one-line reason naming what you see:
{items}

Reply with a JSON object only:
{{"marks": [{{"id": "<item id>", "mark": "yes|no|na", "reason": "<one line>"}}, ...], "note": "<one line on the figure overall>"}}"""


def items_for(request: Dict[str, Any]) -> List[Dict[str, str]]:
    """The general items, then the request's own (``x1``, ``x2``...)."""
    out = [{"id": i, "text": t, "kind": "general"} for i, t in ITEMS]
    out += [{"id": f"x{n}", "text": t, "kind": "request"}
            for n, t in enumerate(request.get("extra") or [], 1)]
    return out


def no_figure_marks(request: Dict[str, Any], why: str = NO_FIGURE
                    ) -> Dict[str, Any]:
    """Every item failed: the run drew nothing (or failed)."""
    return {"marks": {it["id"]: {"mark": "no", "reason": why}
                      for it in items_for(request)},
            "note": why, "no_figure": True, "unmarked": []}


def parse_marks(text: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """The marks in a reply. An item the reply leaves out or marks with
    another word is ``unmarked`` (never ``na``) and listed; a reply that
    marks fewer than half the items is ``not_scored``."""
    try:
        obj = _outer_object(text)
    except Exception:  # noqa: BLE001 - an unreadable reply marks nothing
        obj = {}
    got: Dict[str, Dict[str, str]] = {}
    for m in obj.get("marks") or []:
        if not isinstance(m, dict):
            continue
        mark = str(m.get("mark", "")).strip().lower().replace("/", "")
        mark = {"n.a.": "na", "n/a": "na", "not applicable": "na"}.get(
            mark, mark)
        if mark in MARKS:
            got[str(m.get("id", "")).strip()] = {
                "mark": mark, "reason": str(m.get("reason", ""))[:400]}
    marks, unmarked = {}, []
    for it in items_for(request):
        if it["id"] in got:
            marks[it["id"]] = got[it["id"]]
        else:
            unmarked.append(it["id"])
            marks[it["id"]] = {"mark": UNMARKED,
                               "reason": "the checker gave no mark"}
    out = {"marks": marks, "note": str(obj.get("note", ""))[:400],
           "no_figure": False, "unmarked": unmarked}
    if 2 * len(unmarked) > len(marks):
        out.update(not_scored=True, reply=str(text)[:500],
                   note="the vision reply could not be read: not scored")
    return out


def build_prompt(request: Dict[str, Any], reply: str,
                 numbers: Optional[Dict[str, Any]], names: Sequence[str]
                 ) -> str:
    items = "\n".join(f"- {it['id']}: {it['text']}"
                      for it in items_for(request))
    nums = (json.dumps(numbers, indent=1, default=str)[:4000] if numbers
            else "(none recorded: mark numbers_match 'na')")
    return PROMPT.format(request=request.get("prompt") or request["ask"],
                         numbers=nums, reply=(reply or "")[:3000],
                         n_figures=len(names), names=", ".join(names),
                         items=items)


def image_detail(model: Any, detail: Optional[str] = "auto"
                 ) -> Optional[str]:
    """The image ``detail`` to send: AUTO for a GPT model (HIGH caps the
    image near 768 px on the hosted route), none for Claude."""
    return None if imaging.family_of(model) == "anthropic" else detail


def is_look_copy(path: str) -> bool:
    """Whether ``path`` is a look copy (:data:`LOOK_DIR`)."""
    parts = str(path).replace("\\", "/").split("/")
    return LOOK_DIR in parts[:-1] or parts[-1].lower().endswith(LOOK_SUFFIX)


def run_pngs(paths: Sequence[str]) -> List[str]:
    """A run's PNGs without the look copies, oldest first."""
    pngs = [p for p in paths if str(p).lower().endswith(".png")
            and not is_look_copy(p) and os.path.isfile(p)]
    pngs.sort(key=lambda p: (os.path.getmtime(p), p))
    return pngs


def pick_figures(paths: Sequence[str]) -> List[str]:
    """The newest :data:`MAX_FIGURES` PNGs, look copies left out."""
    return run_pngs(paths)[-MAX_FIGURES:]


def choose_figures(paths: Sequence[str], reply: str = ""
                   ) -> Tuple[List[str], str]:
    """The figures to mark and how they were chosen: ``reply`` when the
    agent's reply names them by file name (the final figures it presents;
    a draft retired with replaces= is deleted and a redraw under the same
    name overwrites the file, but a draft drawn under another name stays
    in ``files/``), else ``newest``."""
    text = (reply or "").lower()
    pngs = run_pngs(paths)
    named = [p for p in pngs if os.path.basename(p).lower() in text]
    if named:
        return named[-MAX_FIGURES:], "reply"
    return pngs[-MAX_FIGURES:], "newest"


def check_run(vision_model: Any, request: Dict[str, Any], run: Dict[str, Any],
              figure_paths: Sequence[str], detail: Optional[str] = "auto"
              ) -> Dict[str, Any]:
    """The vision model's marks for one run's figures."""
    from langchain_core.messages import HumanMessage
    if run.get("error"):
        return dict(no_figure_marks(request, f"the run failed: "
                                    f"{run['error']}"[:300]),
                    run_failed=True)
    shown, how = choose_figures(figure_paths, run.get("answer", ""))
    if not shown:
        return no_figure_marks(request)
    ref = request.get("reference") or {}
    names = [os.path.basename(p) for p in shown]
    text = build_prompt(request, run.get("answer", ""), ref.get("numbers"),
                        names)
    det = image_detail(vision_model, detail)
    limits = imaging.limits_for(vision_model)
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for path in shown:
        img = imaging.prepare_image(imaging.open_image(path), limits)
        url: Dict[str, Any] = {"url": imaging.data_uri(imaging.png_bytes(
            img))}
        if det:
            url["detail"] = det
        content.append({"type": "image_url", "image_url": url})
    reply = vision_model.invoke([HumanMessage(content=content)])
    out = parse_marks(_text_of(getattr(reply, "content", "")), request)
    usage = getattr(reply, "usage_metadata", None) or {}
    out.update(figures=names, picked_by=how, dropped=max(0, len(
        [p for p in figure_paths if p.lower().endswith(".png")
         and not is_look_copy(p)]) - len(shown)),
        detail=det, usage={k: usage.get(k) for k in ("input_tokens",
                                                     "output_tokens")})
    return out
