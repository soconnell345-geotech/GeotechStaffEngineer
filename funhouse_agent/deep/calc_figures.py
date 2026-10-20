"""The calc helper's figure parts (the geotech page's).

:class:`FigureLooks` books a figure a module method drew and asks the calc
helper to look at it; :data:`CALC_REDRAW_NOTE` is added to the calc
helper's description. Moved out of :mod:`funhouse_agent.deep.figure_helper`
(INF-30 step 5), which both pages use: the calc helper, and the module
calls these read, are GeotechStaffEngineer's.
"""

from __future__ import annotations

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - older layout
    from langchain.agents.middleware.types import AgentMiddleware

from funhouse_agent.deep.figure_helper import (MODULE_TOOL, _PICTURES, _base,
                                               _json_of, _picture_paths,
                                               looked_at)

#: Added to the calc helper's description (what the delegating agent reads)
#: when ``plot_code`` is offered (live wave w12, F55j: a second calc helper
#: told to save a figure as ``_v2`` had no code and redrew it from a look).
CALC_REDRAW_NOTE = (
    " To change a figure it drew, name the figure's file in the brief and "
    "keep that name: it is handed the code that drew the figure and "
    "redraws it in place.")


# ---------------------------------------------------------------------------
# On the calc helper: the figures a module method draws are looked at
# ---------------------------------------------------------------------------

#: What a figure a module method drew asks for, as plot_code's ``next``
#: does.
MODULE_FIGURE_LOOK = (
    "LOOK at {files} with analyze_image in a step after this one, before it "
    "goes in a report: ask first what it SHOWS against the data it was "
    "drawn from (two or three plotted values and the axis ranges read back "
    "off it, compared here with the data; nothing drawn that the data do "
    "not give), then the lettering (overlaps, units, scale).")


class FigureLooks(AgentMiddleware):
    """On the calc helper, behind the figure switches (live wave w12, F18j:
    a calc package's five figures went out with no look; BACKLOG GM-151):
    a figure a module method drew (``profile_figure``, ``calc_package``'s
    ``render_figures``, a package's own figures) is booked as a draw of that
    figure this turn, with its plot_data arguments kept beside it, and its
    result asks for a LOOK as plot_code's does; ``html_to_pdf``'s result
    names the figures it embedded that no look in this helper run read.
    Put it just inside :class:`~funhouse_agent.deep.delivered_log.
    DeliveredToolResults`, so the activity log shows its notes (the helper's
    record, outside both, then holds the result with the note after it;
    :func:`_json_of` reads past it)."""

    @staticmethod
    def _note(request, result) -> str:
        import os
        from funhouse_agent import plot_box
        from funhouse_agent._fileio import host_output_dir
        from funhouse_agent.deep.keep_work import (_result_messages,
                                                   current_helper)
        from funhouse_agent.deep.tool_guards import _text
        from funhouse_agent.dispatch import parameters_object
        call = getattr(request, "tool_call", None) or {}
        if call.get("name") != MODULE_TOOL:
            return ""
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        method = str(args.get("method") or "")
        msgs = [m for m in _result_messages(result)
                if getattr(m, "status", None) != "error"]
        res = _json_of("\n".join(_text(m.content) for m in msgs))
        if res is None or res.get("error"):
            return ""
        if method == "html_to_pdf":
            seen = looked_at(current_helper())
            unseen = list(dict.fromkeys(
                _base(p) for p in res.get("embedded_images") or []
                if isinstance(p, str) and p.lower().endswith(_PICTURES)
                and _base(p) not in seen))
            if not unseen:
                return ""
            return ("FIGURES NOT LOOKED AT: this PDF embeds "
                    + ", ".join(unseen) + ", which no look in this helper "
                    "run has read. Look at each with analyze_image (what it "
                    "shows against the data, then the lettering); if a look "
                    "finds a fault, redraw the figure and build the PDF "
                    "again.")
        # a module names its files relative to the working folder
        root = host_output_dir()
        pictures = []
        for p in _picture_paths(res):
            if not os.path.isabs(p) and root:
                p = os.path.join(root, p)
            if os.path.isfile(p) and p not in pictures:
                pictures.append(p)
        if not pictures:
            return ""
        agent = str(args.get("agent_name") or "")
        # a string holding the object runs as it (GM-189)
        params = parameters_object(args.get("parameters"))
        plot_data = agent == "profile_figure" and method == "plot_data"
        booked = plot_box.figures.record_draw(
            pictures, "plot_data" if plot_data else f"{agent}.{method}",
            root, args=params if plot_data and isinstance(params, dict)
            else None)
        note = MODULE_FIGURE_LOOK.format(files=", ".join(
            dict.fromkeys(_base(p) for p in pictures)))
        draw = booked["draw"]
        if draw is not None and draw >= plot_box.DRAW_CAP:
            note += " " + plot_box.draw_cap_note(draw)
        elif draw is not None:
            note += (" Redraw only to fix what the look found, under the "
                     f"same name (this was draw {draw} of at most "
                     f"{plot_box.DRAW_CAP} of this figure this turn).")
        for extra in plot_box.booked_notes(booked):
            note += " " + extra[0].upper() + extra[1:] + "."
        kept = [f["code_file"] for f in booked["figures"]
                if f.get("code_file")]
        if kept:
            note += " Its arguments are kept in " + ", ".join(kept) + "."
        return note

    def wrap_tool_call(self, request, handler):
        from funhouse_agent.deep.keep_work import _with_note
        result = handler(request)
        try:
            note = self._note(request, result)
        except Exception:                                  # noqa: BLE001
            note = ""
        return _with_note(result, note) if note else result

    async def awrap_tool_call(self, request, handler):
        from funhouse_agent.deep.keep_work import _with_note
        result = await handler(request)
        try:
            note = self._note(request, result)
        except Exception:                                  # noqa: BLE001
            note = ""
        return _with_note(result, note) if note else result


__all__ = ["CALC_REDRAW_NOTE", "MODULE_FIGURE_LOOK", "FigureLooks"]
