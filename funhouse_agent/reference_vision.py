"""GSE's own readers of its reference library, beside the core's vision tools.

Moved out of ``vision_tools.py`` when that file went into the shared core
(BACKLOG INF-32, the app split's step 11 (a); plan
``module_work/APP_SPLIT_PLAN.md`` sections 2.3 and 2.7): the core imports
nothing of GeotechStaffEngineer, and these read GSE's reference library
(``geotech_references``, :mod:`funhouse_agent.reference_docs`,
:mod:`funhouse_agent.worked_examples`, :mod:`funhouse_agent.chart_reading`).
The code is as it was; it reaches the core's vision tools through the
module (``_vt.ask_vision``...), so a test that patches them by either name
still reaches these readers.

GSE hands them to the core as hooks (:mod:`aec_agent_core.hooks`), filled
in ``funhouse_agent/__init__.py``:

- ``vision_tools.reference_read_roots`` -> :func:`reference_read_roots`
  (the folders the read tools may also read);
- ``vision_tools.read_reference_figure`` ->
  :func:`_dispatch_read_reference_figure`;
- ``vision_tools.view_worked_example_source`` ->
  :func:`_dispatch_view_worked_example`.
"""

import json
import os
import re
from typing import List

from aec_agent_core import vision_tools as _vt


def reference_read_roots() -> List[str]:
    """GSE's reference library folders, after ``GEOTECH_REFERENCES_DOCS``
    (which the core reads itself): the cache the app fetches reference PDFs
    into, the installed library's ``docs``, a source checkout's
    ``geotech-references/docs`` and each local reference pack's folder
    (REF-64; none with no packs)."""
    roots: List[str] = []
    try:
        from funhouse_agent import reference_docs
        roots.append(reference_docs.cache_dir())
    except Exception:  # noqa: BLE001
        pass
    try:
        from geotech_references import _figures_db
        roots.append(os.path.join(str(_figures_db._REPO_ROOT), "docs"))
    except Exception:  # noqa: BLE001
        pass
    # This file sits where vision_tools.py did (funhouse_agent/), so the
    # checkout's folder is the same one it named.
    roots.append(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "geotech-references", "docs"))
    # A pack's PDF may sit in its own folder: a read-off names it by its file
    # name there, and render_region / read_document then find it.
    from funhouse_agent.reference_packs import pack_folders
    roots += pack_folders()
    return roots


_READ_OFF_NOTE = (
    "Value(s) are a vision read-off estimate from the chart — accurate to a few "
    "percent on linear axes, looser on log axes or dense curve families. Verify "
    "against a closed-form or digitized method where one exists."
)

#: The note on a reference chart read-off that also carries code's reading
#: (W3 step 7): which value rests on what.
_READ_OFF_NOTE_WITH_CODE = (
    "Two readings. 'analysis' is a vision read-off estimate from the chart "
    "image (by eye: a few percent on linear axes, looser on log axes or "
    "dense curve families). 'code_reading' is measured from the chart's own "
    "drawing - "
    "its axes fitted to the gridlines and printed labels, the curve found "
    "where it crosses the input - with its own +/-: where it gives a value, "
    "that is the value to use, with the vision value beside it; one marked "
    "'disagree' differs by more than about 3x code's +/-, so check it before "
    "relying on either. Where code gives no value it says why, and the "
    "vision read-off stands alone. Verify against a closed-form or digitized "
    "method where one exists."
)


def _dispatch_view_worked_example(arguments, engine):
    """Chart read-off: rendered and sent at the chart budget
    (``vision_view.chart_reading``) — see ``_dispatch_view_worked_example_at_budget``."""
    from aec_agent_core import vision_view
    with vision_view.chart_reading():
        return _dispatch_view_worked_example_at_budget(arguments, engine)


def _dispatch_view_worked_example_at_budget(arguments, engine):
    """Render a worked example's printed source page and analyze it via vision.

    The corpus (funhouse_agent/worked_examples.json) catalogues, per entry with
    a source PDF in the docs folder, ``source_doc`` + 1-based
    ``source_pdf_pages`` (page-search located — treated as estimated). This is
    the sample-calc twin of ``read_reference_figure``: same PDF resolution
    convention (GEOTECH_REFERENCES_DOCS / repo docs), same 220-dpi render, same
    read-off caveat.
    """
    example_id = arguments.get("example_id", "")
    pdf_page = arguments.get("pdf_page")           # 1-based; optional
    question = (arguments.get("prompt", "")
                or "Describe the worked example on this page: the given "
                   "values, the calculation steps shown, and any figures.")

    if not example_id:
        return json.dumps({"error": "'example_id' is required "
                                    "(find one via find_worked_examples)."})
    try:
        from funhouse_agent import worked_examples as _we
        entry = _we.get_example(example_id)
        if entry is None:
            known = [e.get("id") for e in _we.load_examples()]
            return json.dumps({"error": f"Unknown worked example "
                                        f"'{example_id}'. Known ids: {known}"})
        pages = entry.get("source_pdf_pages") or []
        if not pages and pdf_page is None:
            return json.dumps({
                "error": f"Worked example {example_id} has no catalogued "
                         "source pages (source PDF not in the docs folder — "
                         "e.g. the Slide2 verification manual). Its problem/"
                         "dispatch_calls text is still available via "
                         "get_worked_example."})
        page_1b = int(pdf_page) if pdf_page is not None else int(pages[0])
        try:
            pdf_abs = _we.resolve_source_pdf(entry)
        except FileNotFoundError as e:
            from funhouse_agent import reference_docs
            pdf_abs = reference_docs.fetch(entry.get("source_doc", ""))
            if not pdf_abs:
                return json.dumps({"error": reference_docs.missing_message(
                    str(e), entry.get("source_doc", ""))})
    except (KeyError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    from aec_agent_core import vision_view
    try:
        image_bytes, info = vision_view.render_view(
            str(pdf_abs), page=page_1b - 1,
            allow_jpeg=getattr(engine, "accepts_jpeg", False),
            engine=engine)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"})
    except ValueError as e:
        return json.dumps({"error": str(e)})

    framing = (
        f"This is page {page_1b} (of the pages catalogued for worked example "
        f"{entry['id']}: {pages or [page_1b]}) from \"{entry.get('source', '')}\" "
        f"— the printed source of: {entry.get('title', '')}. The catalogued "
        "page numbers were located by text search and may be off by a page — "
        "if this page does not show the expected content, say so plainly. "
        f"Task: {question}"
    )
    try:
        analysis = _vt.ask_vision(
            engine, image_bytes,
            vision_view.with_location(framing, _vt._sent_size(info)),
            "worked_example_page")
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({"error": f"Vision not available on this engine: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    out = {
        "example_id": entry["id"],
        "source_doc": entry.get("source_doc"),
        "pdf_page": page_1b,
        "catalogued_pages": pages,
        "analysis": analysis,
        "note": _READ_OFF_NOTE,
        # The PDF by its name in the reference library, which render_region
        # resolves; the server path stays internal.
        "source": _vt.display_path(str(pdf_abs)),
        "page": page_1b - 1,
        **vision_view.view_payload(info, engine),
    }
    _vt._finish_answer(out, info, tagged_only=True)
    return json.dumps(out)


#: The heading the catalog's ``read_notes`` go under in the read prompt.
READ_NOTES_HEADING = ("Notes from the reference's catalog on reading this "
                      "figure")
#: The most of a figure's ``read_notes`` the read prompt carries: the limit
#: the library's ``packs check`` holds a catalog to
#: (``geotech_references.packs.READ_NOTES_MAX``), kept here for a catalog
#: that was never checked.
READ_NOTES_MAX = 500


#: The step every chart read starts with (2026-11-10 reference review, item
#: 6): it confirms the figure asked for and states the axes it sees, which
#: had been asked only of a page located by estimate.
_CONFIRM_STEP = (
    "1. FIRST confirm you are looking at Figure {fig}: give the figure "
    "number and caption as printed on the page, and the quantity and unit "
    "printed on each axis. If Figure {fig} is not on this page, say so "
    "plainly and do not read a value{estimated}. Ignore other figures and "
    "body text.\n")
_CONFIRM_ESTIMATED = (" (the page lookup was estimated, so an adjacent page "
                      "should be tried)")

#: Said after the steps: what is the page's and what is the reader's own.
_PRINTED_ONLY = (
    "Report only what is printed on this page. Mark anything else as your "
    "own and say what it rests on: a value from memory, a unit conversion, "
    "a curve extended past its printed end.\n\n")

#: How long the reason is when code's measurement gave no value.
_NO_VALUE_REASON_CHARS = 160

#: Said beside the read when the library computes the same figure.
_DIGITIZED_NOTE = (
    "The library also computes this figure with the method(s) named here "
    "(call_agent on the module; not called by this read). Call one with the "
    "same inputs and compare: where the two differ by more than this read's "
    "stated tolerance, report both values.")


def _figure_tokens(figure_number: str) -> List[str]:
    """A figure number as it appears in method names: '7-15' -> '7_15',
    '4.3' -> '4_3', '5-18(a)' -> '5_18_a' and '5_18a'."""
    tok = "_".join(t for t in re.split(
        r"[^a-z0-9]+", str(figure_number or "").lower()) if t)
    if not tok:
        return []
    joined = re.sub(r"_(?=[a-z]$)", "", tok)
    return list(dict.fromkeys((tok, joined)))


def same_figure_methods(catalog_id: str, figure_number: str,
                        catalog_title: str = ""):
    """The reference module serving ``catalog_id`` and its methods that
    compute ``figure_number`` (their names carry the figure number:
    ``figure_7_15_...``, a lettered part ``figure_5_18a_...`` counting as
    its figure), as ``(module, [methods])``; drawing methods
    (``plot_...``) are left out. A module that holds more than one manual
    keeps only methods whose own reference names this catalog's document
    number (DM7.1's 'Figure 5-16' is not DM7.2's)."""
    from funhouse_agent import dispatch
    module = dispatch.reference_module(catalog_id)
    tokens = _figure_tokens(figure_number)
    if module is None or not tokens:
        return module, []
    try:
        info = dispatch._load_adapter(module).METHOD_INFO
    except Exception:  # noqa: BLE001 - a pointer, never the read
        return module, []
    # a lettered part counts as its figure: figure_5_18a is Figure 5-18
    pats = [re.compile(rf"(?:^|_)figures?_{re.escape(t)}[a-z]?(?:_|$)")
            for t in tokens]
    methods = [name for name in info
               if not name.startswith("plot_")
               and any(p.search(name) for p in pats)]
    if methods:
        try:
            from geotech_references import _figures_db
            served = {r["reference"] for r in _figures_db.list_indexed_figures()
                      if dispatch.reference_module(r["reference"]) == module}
        except Exception:  # noqa: BLE001
            served = set()
        numbers = re.findall(r"\d+(?:-\d+)+", str(catalog_title or ""))
        if len(served) > 1 and numbers:
            methods = [name for name in methods if any(
                n in f"{info[name].get('reference', '')} "
                     f"{info[name].get('references', '')}" for n in numbers)]
    return module, methods


def _no_code_value(block: dict) -> str:
    """The plain sentence for a chart read whose code measurement gave no
    value (in the recorded reads, 22 of 22)."""
    reason = block.get("note") or ""
    for row in block.get("readings") or []:
        if reason:
            break
        reason = row.get("note") or row.get("status") or ""
    reason = " ".join(str(reason or block.get("status") or "").split())
    if len(reason) > _NO_VALUE_REASON_CHARS:
        reason = reason[:_NO_VALUE_REASON_CHARS].rsplit(" ", 1)[0] + "..."
    return ("Code's measurement of the chart gave no value"
            + (f" ({reason})" if reason else "")
            + ": the value here is the vision read-off alone, not checked "
              "by a second reading.")


def _dispatch_read_reference_figure(arguments, engine):
    """Chart read-off: rendered and sent at the chart budget
    (``vision_view.chart_reading``) — see ``_dispatch_read_reference_figure_at_budget``."""
    from aec_agent_core import vision_view
    with vision_view.chart_reading():
        return _dispatch_read_reference_figure_at_budget(arguments, engine)


def _dispatch_read_reference_figure_at_budget(arguments, engine):
    """Render a catalogued reference figure and read value(s) off it via vision."""
    reference = arguments.get("reference", "")
    figure_number = arguments.get("figure_number", "")
    question = arguments.get("prompt", "") or "Read the relevant value(s) from this chart."

    if not reference or not figure_number:
        return json.dumps({
            "error": "Both 'reference' and 'figure_number' are required "
                     "(find them via figure_db.figure_search)."
        })

    # Resolve the figure to its source PDF page.
    try:
        from geotech_references import _figures_db
        rec = _figures_db.figure_get(reference, figure_number)
        pdf_abs, page_idx = _figures_db.resolve_pdf(reference, figure_number)
    except KeyError as e:
        return json.dumps({"error": f"Figure not found: {e}"})
    except FileNotFoundError as e:
        # Not in the local docs folder: a host-registered source (the web
        # app's SharePoint primary_references/) fetches it on first use.
        from funhouse_agent import reference_docs
        pdf_abs = reference_docs.fetch(rec.get("pdf_path", ""))
        if not pdf_abs:
            return json.dumps({"error": reference_docs.missing_message(
                str(e), rec.get("pdf_path", ""))})
        page_idx = int(rec["pdf_page_index"])
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # Render the page at the vision model's image budget: the largest image
    # it reads without shrinking, so curves and axis labels stay legible.
    # A catalog may box the chart on its page (``bbox``, PDF points, top-left
    # origin; REF-64): the chart, not the whole page, then fills the image
    # budget. Built-in figures carry no box, so their render is as before.
    render_kwargs = {}
    if rec.get("bbox"):
        render_kwargs["bbox"] = [float(v) for v in rec["bbox"]]
    from aec_agent_core import vision_view
    try:
        image_bytes, info = vision_view.render_view(
            str(pdf_abs), page=page_idx,
            allow_jpeg=getattr(engine, "accepts_jpeg", False),
            engine=engine, **render_kwargs)
    except ImportError:
        return json.dumps({
            "error": "PyMuPDF required for PDF rendering. pip install PyMuPDF"
        })
    except ValueError as e:
        return json.dumps({"error": str(e)})

    # Some catalog pages are estimated (not caption-confirmed); warn the vision
    # model so it verifies the figure is present instead of reading a wrong page.
    estimated = rec.get("page_estimated")
    locate_clause = "that should contain" if estimated else "containing"
    est_caveat = (
        f" NOTE: the page for Figure {rec['figure_number']} was located by "
        "ESTIMATE and may be off by a page or two."
        if estimated else ""
    )
    step_one = _CONFIRM_STEP.format(
        fig=rec["figure_number"],
        estimated=_CONFIRM_ESTIMATED if estimated else "")
    full_prompt = (
        f"This image is a rendered page from {reference} {locate_clause} Figure "
        f"{rec['figure_number']}: \"{rec['caption']}\".{est_caveat}\n\n"
        "Read the requested value(s) off this engineering chart:\n"
        + step_one +
        "2. Identify the axes (note any logarithmic scales) and the family of "
        "curves and what parameter distinguishes them.\n"
        "3. For the requested inputs, select the correct curve (interpolating "
        "between curves where needed) and read the value at the right axis "
        "position.\n"
        "4. Report the value(s) clearly with the unit printed on the axis, "
        "state which curve/axis you used, and flag that this is a chart "
        "read-off estimate.\n\n"
        + _PRINTED_ONLY +
        f"Request: {question}"
    )
    # The catalog's own notes on reading this figure (``read_notes``,
    # REF-64), at most READ_NOTES_MAX characters. Built-in figures carry
    # none, so their prompt is as before.
    notes = str(rec.get("read_notes") or "").strip()[:READ_NOTES_MAX].rstrip()
    if notes:
        full_prompt += f"\n\n{READ_NOTES_HEADING}:\n{notes}"
    # The second voter (W3 step 7): the side call also says, per curve it
    # read, where on the image the curve crosses the input, so code can
    # measure the same thing from the drawing.
    from funhouse_agent import chart_reading
    code_voter = chart_reading.voter_on()
    if code_voter:
        full_prompt = chart_reading.with_read_lines(full_prompt)

    try:
        result = _vt.ask_vision(
            engine, image_bytes,
            vision_view.with_location(full_prompt, _vt._sent_size(info)),
            "reference_figure")
    except (NotImplementedError, AttributeError) as e:
        return json.dumps({"error": f"Vision not available on this engine: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    out = {
        "reference": reference,
        "figure_number": rec["figure_number"],
        "caption": rec["caption"],
        # the page as a PDF viewer numbers it (1-based), beside the 0-based
        # index the renderer takes (2026-11-10 reference review, item 6)
        "pdf_page": page_idx + 1,
        "pdf_page_index": page_idx,
        "page_estimated": rec.get("page_estimated", False),
        "analysis": result,
        "note": _READ_OFF_NOTE,
        # Zooming on the chart (an axis, a curve label) goes through
        # render_region on the same PDF page -- named by its file name in
        # the reference library, which render_region resolves.
        "source": _vt.display_path(str(pdf_abs)),
        "page": page_idx,
        **vision_view.view_payload(info, engine),
    }
    if code_voter:
        # Read off the RAW answer, before its pixel boxes are rewritten on
        # the 0-999 grid for the agent: code places them with the size sent.
        try:
            block = chart_reading.code_reading(
                str(pdf_abs), page_idx, str(result or ""), info["clip"],
                _vt._sent_size(info), engine)
        except Exception as e:  # noqa: BLE001 - a second voter, never the answer
            block = {"status": "unavailable",
                     "note": f"code could not read the chart: "
                             f"{type(e).__name__}: {e}"}
        out["code_reading"] = block
        out["note"] = _READ_OFF_NOTE_WITH_CODE
        if block.get("status") != "measured":
            # said at the top, in plain words (item 6: 22 of 22 recorded
            # reads had no code value, said only inside the block)
            out["second_reading"] = _no_code_value(block)
    # The library's own function for this figure, named, not called: the
    # helper calls it and compares (item 6).
    try:
        module, methods = same_figure_methods(
            rec.get("reference") or reference, rec["figure_number"],
            rec.get("reference_title") or "")
    except Exception:  # noqa: BLE001 - a pointer, never the read
        module, methods = None, []
    if methods:
        out["digitized_lookup"] = {"module": module, "methods": methods,
                                   "note": _DIGITIZED_NOTE}
    _vt._finish_answer(out, info, tagged_only=True)
    return json.dumps(out)
