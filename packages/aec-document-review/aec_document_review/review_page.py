"""The Document Review page's own parts, in one file (INF-31, step 7; copied
into this package at the app split's step 13, BACKLOG INF-33).

Since the copy: the page shows its own notice (:mod:`.notice`, a
placeholder until OWN-78), runs no judgment-call check at a turn's end
(GeotechStaffEngineer's; the review build never asked for a consult, so it
only recorded), has no reference-PDF line in its diagnostics (no reference
library here), and names its own distribution. The text below is the
original's.

Its profile (:data:`DOCUMENT_REVIEW`), the orientation turn a fresh upload
gets, the page's agent builder, and its notice. When Document Review gets
its own copy of the page (the app split's step 13), this file goes with it
and ``aec_document_review/geotech_page.py`` stays behind. :mod:`aec_document_review.profiles`
re-exports every name here, so ``profiles.DOCUMENT_REVIEW`` and the
orientation names keep working.

Until step 13 the page keeps three things it shares with the geotech page:
the current professional-use notice (the owner has not approved a new
wording, OWN-78), the judgment-call check at each turn's end, and the
reference-PDF line in the connection diagnostics. Each is reached inside a
``try`` or by name on use, never when this file loads.
"""

from __future__ import annotations

from typing import Optional

from aec_document_review.page_profile import AppProfile

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
    :class:`aec_document_review.core.Attachment` objects) — the text the app and the
    review suite both send: the profile's own request with the names filled
    in. (The page-count variants behind ``GEOTECH_REVIEW_OVERVIEW`` were
    removed in BACKLOG DR-39: the suite measured no gain.)
    """
    atts = list(attachments or [])
    names = ", ".join(f"`{a.key}`" for a in atts)
    return profile.orientation_request.format(names=names)


def finish_build_kwargs(kw: dict) -> dict:
    """The page's build arguments with its prompt: the document-review one,
    built at agent-build time so the memory flag can be honoured."""
    if kw.pop("_document_review_prompt", False):
        from aec_document_review.review_prompt import (
            build_document_review_prompt)
        kw["system_prompt"] = build_document_review_prompt(
            memory_enabled=bool(kw.get("enable_memory")))
    return kw


def build_page_agent(model, *, attachments, save_fn, temp_dir, **kw):
    """The Document Review page's agent, after the shell's shared wiring
    (:func:`aec_document_review.core.build_agent`): the page's own build
    (:func:`aec_document_review.review_build.build_review_page_agent`, INF-31
    step 8)."""
    # THIS conversation's folder, bound when the agent is built (the
    # process-wide working folder can be re-pointed by another tab): the
    # minimal build records it; the default build accepts and ignores it.
    kw.setdefault("working_dir", temp_dir)
    from aec_document_review.review_build import build_review_page_agent
    return build_review_page_agent(model, attachments=attachments,
                                   save_fn=save_fn, **kw)


def disclaimer_text() -> Optional[str]:
    """The professional-use notice the page shows (:mod:`.notice`)."""
    try:
        from aec_document_review.notice import notice_text
        return notice_text() or None
    except Exception:  # noqa: BLE001 - the shell has a fallback
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
        # Marks the page's build (aec_document_review.review_build since step 8;
        # build_deep_agent hands a review_page build there too).
        "review_page": True,
    },
    finish_build_kwargs=finish_build_kwargs,
    disclaimer=disclaimer_text,
    placeholder=("Ask about the document — or ask for a marked-up copy or a "
                 "Word summary…"),
    agent_builder=build_page_agent,
    version_dists=("aec-document-review",),
)


__all__ = ["DOCUMENT_REVIEW", "ORIENTATION_REQUEST",
           "IMAGE_SUFFIXES", "orientation_request_for",
           "orient_on_upload", "build_page_agent", "disclaimer_text",
           "finish_build_kwargs"]
