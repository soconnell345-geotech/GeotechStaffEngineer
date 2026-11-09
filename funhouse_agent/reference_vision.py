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
from typing import List

from aec_agent_core import vision_tools as _vt


def reference_read_roots() -> List[str]:
    """GSE's reference library folders, after ``GEOTECH_REFERENCES_DOCS``
    (which the core reads itself): the cache the app fetches reference PDFs
    into, the installed library's ``docs`` and a source checkout's
    ``geotech-references/docs``."""
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
    from aec_agent_core import vision_view
    try:
        image_bytes, info = vision_view.render_view(
            str(pdf_abs), page=page_idx,
            allow_jpeg=getattr(engine, "accepts_jpeg", False),
            engine=engine)
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
    step_one = (
        f"1. FIRST confirm Figure {rec['figure_number']} actually appears on this "
        "page. If it does NOT, say so plainly and do not read a value — report "
        "that the page lookup was estimated and an adjacent page should be tried.\n"
        if estimated else
        f"1. Locate Figure {rec['figure_number']} on the page; ignore other "
        "figures and body text.\n"
    )
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
        "4. Report the value(s) clearly, state which curve/axis you used, and "
        "flag that this is a chart read-off estimate.\n\n"
        f"Request: {question}"
    )
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
    _vt._finish_answer(out, info, tagged_only=True)
    return json.dumps(out)
