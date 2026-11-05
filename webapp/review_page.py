"""The Document Review page's own parts, in one file (INF-31, step 7).

Its profile (:data:`DOCUMENT_REVIEW`), the orientation turn a fresh upload
gets, the page's agent builder, and its notice. When Document Review gets
its own copy of the page (the app split's step 13), this file goes with it
and ``webapp/geotech_page.py`` stays behind. :mod:`webapp.profiles`
re-exports every name here, so ``profiles.DOCUMENT_REVIEW`` and the
orientation names keep working.

Until step 13 the page keeps three things it shares with the geotech page:
the current professional-use notice (the owner has not approved a new
wording, OWN-78), the judgment-call check at each turn's end, and the
reference-PDF line in the connection diagnostics. Each is reached inside a
``try`` or by name on use, never when this file loads.
"""

from __future__ import annotations

import os
from typing import Optional

from webapp.page_profile import AppProfile

#: The request the shell sends on the user's behalf after an upload, when the
#: profile asks for an orientation turn. It reads as the user's own message
#: in the transcript, and it asks for a bearing, not a review.
ORIENTATION_REQUEST = (
    "I just attached {names}. Before I ask anything, give me a short "
    "orientation: open it, take in the page map and the structure (and the "
    "contact sheets if it is long), then tell me — in under 250 words — what "
    "this appears to be; how it is organised, with the printed page or sheet "
    "numbers where the document has them; what you can read as text and what "
    "you would have to look at; any existing reviewer markups (how many, by "
    "whom); and two or three things you could do next with it. Do not review "
    "it yet."
)

#: What the orientation asks for once the bearing is taken (shared by the
#: two ``GEOTECH_REVIEW_OVERVIEW`` variants; the same words close
#: :data:`ORIENTATION_REQUEST`).
_ORIENTATION_ASK = (
    "tell me — in under 250 words — what this appears to be; how it is "
    "organised, with the printed page or sheet numbers where the document has "
    "them; what you can read as text and what you would have to look at; any "
    "existing reviewer markups (how many, by whom); and two or three things "
    "you could do next with it. Do not review it yet."
)

#: ``GEOTECH_REVIEW_OVERVIEW``: the uploads run past the page threshold, so
#: the contact sheets (one per 48 pages) are asked for by name — good for
#: finding the plans, logs, tables and marked-up pages of a long document.
ORIENTATION_REQUEST_LONG = (
    "I just attached {names} ({pages} pages in all). Before I ask anything, "
    "give me a short orientation: open it and take in the page map and the "
    "structure; because it is long, also render the contact sheets of its "
    "pages (render_page_thumbnails) and look at them before you summarise. "
    "Then " + _ORIENTATION_ASK
)

#: ``GEOTECH_REVIEW_OVERVIEW``: at or under the threshold every page can be
#: looked at individually, so the contact sheets are skipped.
ORIENTATION_REQUEST_SHORT = (
    "I just attached {names}. Before I ask anything, give me a short "
    "orientation: open it and take in the page map and the structure. It is "
    "short, so skip the contact sheets, and look at individual pages only "
    "where the text does not tell you what they are. Then " + _ORIENTATION_ASK
)


def pdf_page_count(path: Optional[str]) -> int:
    """Pages in the PDF at ``path``; 0 for anything that is not a readable
    PDF (an image, a spreadsheet, a missing file). Never raises."""
    try:
        with open(os.fspath(path), "rb") as fh:
            head = fh.read(1024)
        if b"%PDF" not in head:
            return 0
        import pymupdf
        doc = pymupdf.open(os.fspath(path), filetype="pdf")
        try:
            return int(doc.page_count)
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 - a count, never an error
        return 0


#: Uploads that are pictures, not documents to review.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif",
                  ".tiff")


def orient_on_upload(keys, transcript) -> bool:
    """Whether an upload gets the page's automatic orientation turn.

    Not when everything uploaded is an image and the conversation has already
    started: a screenshot dropped in mid-conversation is evidence for the
    next message, not a document to be introduced to. Field session
    2026-10-01: the owner attached a screenshot of a missed tag, the app sent
    "Before I ask anything, give me a short orientation…" on their behalf, and
    the agent described the screenshot instead of answering their point. The
    upload note still tells the agent the file is there.
    """
    keys = [str(k) for k in (keys or [])]
    if not keys:
        return False
    images_only = all(k.lower().endswith(IMAGE_SUFFIXES) for k in keys)
    started = any(isinstance(e, dict) and e.get("role") in ("user",
                                                          "assistant")
                  for e in (transcript or []))
    return not (images_only and started)


def orientation_request_for(profile: "AppProfile", attachments) -> str:
    """The orientation request for ``attachments`` (staged
    :class:`webapp.core.Attachment` objects) — the text the app and the
    review suite both send.

    With ``GEOTECH_REVIEW_OVERVIEW`` off it is exactly the profile's own
    request. On, the PDF pages of the new uploads are counted: above
    ``GEOTECH_REVIEW_OVERVIEW_PAGES`` (default 20) in all, the request asks
    for the contact sheets by name and to look at them before summarising;
    at or below it, to skip them and look at pages only where needed.
    """
    atts = list(attachments or [])
    names = ", ".join(f"`{a.key}`" for a in atts)
    try:
        from funhouse_agent import review_flags
        overview = review_flags.overview()
        threshold = review_flags.overview_pages()
    except Exception:  # noqa: BLE001 - no switches, no change
        overview = False
    if not overview:
        return profile.orientation_request.format(names=names)
    pages = sum(pdf_page_count(getattr(a, "path", None)) for a in atts)
    if pages > threshold:
        return ORIENTATION_REQUEST_LONG.format(names=names, pages=pages)
    return ORIENTATION_REQUEST_SHORT.format(names=names)


def finish_build_kwargs(kw: dict) -> dict:
    """The page's build arguments with its prompt: the document-review one,
    built at agent-build time so the memory flag can be honoured."""
    if kw.pop("_document_review_prompt", False):
        from funhouse_agent.review_prompt import (
            build_document_review_prompt)
        kw["system_prompt"] = build_document_review_prompt(
            memory_enabled=bool(kw.get("enable_memory")))
    return kw


def build_page_agent(model, *, attachments, save_fn, temp_dir, **kw):
    """The Document Review page's agent, after the shell's shared wiring
    (:func:`webapp.core.build_agent`). Today's build is still the shared
    builder's ``review_page=True`` branch (the app split's step 8 gives the
    page a build of its own)."""
    # The Document Review page's lean agent binds its findings and digests
    # to THIS conversation's folder when it is built; the process-wide
    # working folder can be re-pointed by another tab. (The legacy build
    # accepts and ignores it.)
    kw.setdefault("working_dir", temp_dir)
    from funhouse_agent.deep.agent import build_deep_agent
    return build_deep_agent(model, attachments=attachments, save_fn=save_fn,
                            **kw)


def disclaimer_text() -> Optional[str]:
    """The professional-use notice the page shows: for now the package's
    current one (OWN-78 decides the page's own wording; it goes at step
    13). None where it cannot be had (the shell then shows its short
    notice)."""
    import io
    try:
        from funhouse_agent import disclaimer
        buf = io.StringIO()
        disclaimer(file=buf)
        return buf.getvalue().strip() or None
    except Exception:  # noqa: BLE001 - the shell has a fallback
        return None


def turn_checks(agent, callbacks):
    """The check of the answer at each turn's end: the judgment-call check,
    as the geotech page runs it (it only records here: the review build
    never asks for a consult). Kept until step 13 so a turn's record is
    unchanged. None where it cannot be had."""
    try:
        from funhouse_agent.judgment_check import (RECORD_EVENT,
                                                   TurnJudgmentCheck,
                                                   mode_for)
        logger = next((cb for cb in callbacks or ()
                       if hasattr(cb, "conv_dir")
                       and hasattr(cb, "on_custom_event")), None)
        log = None
        if logger is not None:
            def log(data, _logger=logger):
                _logger.on_custom_event(RECORD_EVENT, data, run_id=None)
        return TurnJudgmentCheck(mode=mode_for(agent), log=log)
    except Exception:  # noqa: BLE001 - the turn runs without it
        return None


DOCUMENT_REVIEW = AppProfile(
    name="document_review",
    title="Document Review",
    icon="📄",
    caption=("Reads drawings, specifications, submittals, reports and "
             "calculation packages — looking at the pages, not only the text "
             "— and hands back Word memos and marked-up PDFs. A review aid; "
             "the judgement stays with you."),
    url_path="",                 # the default page: served at "/"
    specialists=False,
    orientation=True,
    orientation_request=ORIENTATION_REQUEST,
    _build_kwargs={
        # No analysis modules and no reference library: the four dispatch
        # tools and the two reference sub-agents are left out, and the prompt
        # is the document-review one (built at agent-build time so the memory
        # flag can be honoured).
        "allowed_agents": (),
        "reference_mode": "off",
        # The behavior pickers' geotech presets do not apply here: no calc
        # sub-agent (there are no calculation modules to delegate to) and no
        # analysis-depth prompt preset.
        "enable_calc_subagent": False,
        "extra_system_prompt": None,
        "_document_review_prompt": True,
        # With GEOTECH_REVIEW_AGENT=lean, build_deep_agent hands this build
        # to the page's own agent (funhouse_agent.deep.review_agent); off,
        # it is ignored and the page is built as it has been since 5.26.
        "review_page": True,
    },
    finish_build_kwargs=finish_build_kwargs,
    disclaimer=disclaimer_text,
    placeholder=("Ask about the document — or ask for a marked-up copy or a "
                 "Word summary…"),
    agent_builder=build_page_agent,
    turn_checks=turn_checks,
    # The reference-PDF line, as the geotech page shows it (step 13 drops
    # it from this page); named, so this file never loads the geotech one.
    diagnostics_checks="webapp.geotech_page:diagnostics_checks",
    version_dists=("geotech-staff-engineer",),
)


__all__ = ["DOCUMENT_REVIEW", "ORIENTATION_REQUEST",
           "ORIENTATION_REQUEST_LONG", "ORIENTATION_REQUEST_SHORT",
           "IMAGE_SUFFIXES", "orientation_request_for", "pdf_page_count",
           "orient_on_upload", "build_page_agent", "disclaimer_text",
           "turn_checks", "finish_build_kwargs"]
