"""The page's profile (copied from GeotechStaffEngineer's ``webapp/profiles.py``
at the app split's step 13, BACKLOG INF-33; that app's text follows).

Since the copy this package serves ONE page, Document Review
(:data:`DOCUMENT_REVIEW`, in :mod:`aec_document_review.review_page`), which
is also the default. What the old text says of the geotech page is history.


The owner's rule for the Tiny Apps build (2026-09-21): two pages, NO second
copy of the app. Page one is a general document-review agent for anyone who
designs or builds — the planlens front door; page two is GeotechStaffEngineer
as it has always been. Both run the SAME ``aec_document_review/app.py`` shell, the same
persistence, the same engine, the same file and vision tools. What differs is
an :class:`AppProfile`: the title on the page, the agent the shell builds
(prompt, tool scope, sub-agents), whether the specialist picker applies,
whether a fresh upload gets an automatic orientation turn, the hooks the
shell asks instead of the geotechnical toolkit (:mod:`aec_document_review.page_profile`),
and where that person's conversations for that page live on disk. Each
page's own parts are in a file of its own: ``aec_document_review/geotech_page.py`` and
``aec_document_review/review_page.py`` (INF-31, the app split's step 7).

The current profile is chosen by the page function in
:mod:`aec_document_review.tinyapps_entry` (``st.navigation``), or by the
``GEOTECH_APP_PROFILE`` environment variable for a single-page launch, and
defaults to the geotech profile — so the Databricks launcher and a plain
``streamlit run aec_document_review/app.py`` behave exactly as before.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from aec_document_review import core
from aec_document_review.identity import Identity
from aec_document_review.page_profile import AppProfile
# The Document Review page's own parts (its profile, the orientation turn,
# its builder and notice) are in aec_document_review/review_page.py (INF-31 step 7); the
# names stay importable from here.
from aec_document_review.review_page import (  # noqa: F401
    DOCUMENT_REVIEW, IMAGE_SUFFIXES, ORIENTATION_REQUEST,
    orient_on_upload, orientation_request_for,
)

#: Env var naming the profile for a single-page launch.
PROFILE_ENV = "GEOTECH_APP_PROFILE"
#: The session-state key the entry's page function sets before the shell runs.
SESSION_KEY = "_app_profile"

#: The one page (the Document Review page; GeotechStaffEngineer's page stays
#: in that app since the split's step 13).
PROFILES: Dict[str, AppProfile] = {DOCUMENT_REVIEW.name: DOCUMENT_REVIEW}
DEFAULT = DOCUMENT_REVIEW


def get(name: Optional[str]) -> AppProfile:
    """The profile called ``name``; the default for an unknown or empty name."""
    return PROFILES.get((name or "").strip().lower(), DEFAULT)


def current() -> AppProfile:
    """The profile this script run serves: the page function's choice in
    session state, else ``GEOTECH_APP_PROFILE``, else the geotech default.
    Safe to call outside a Streamlit run (tests, the launcher)."""
    name = None
    try:
        import streamlit as st
        name = st.session_state.get(SESSION_KEY)
    except Exception:
        name = None
    return get(name or os.environ.get(PROFILE_ENV))


def set_current(name: str) -> AppProfile:
    """Record the page's profile in session state (the entry calls this
    before running the shell)."""
    import streamlit as st
    prof = get(name)
    st.session_state[SESSION_KEY] = prof.name
    return prof


def session_root(profile: AppProfile, identity: Identity,
                 base: Optional[str] = None) -> str:
    """Where this person's conversations for this page live.

    On a MULTI-USER host (the caller named by the proxy header):
    ``<data root>/users/<identity key>/<profile>``. In a single-user process
    — nobody identified, ``DEV_IDENTITY``, or the Databricks launcher's
    email — the geotech page keeps the data root itself, the layout every
    existing deployment has, so the Databricks app and a local ``streamlit
    run`` keep finding the conversations they already have; the review page
    gets its own folder beside it (``<data root>/pages/document_review``) so
    the two pages' lists never mix.
    """
    root = base or core.data_root()
    if identity.multi_user:
        return os.path.join(root, "users", identity.key, profile.name)
    # Since the copy (step 13) the page is this package's default, but it
    # keeps the folder it had beside the geotech page, so a single-user
    # host keeps finding its conversations.
    return os.path.join(root, "pages", profile.name)


__all__ = ["AppProfile", "DOCUMENT_REVIEW", "PROFILES", "DEFAULT",
           "ORIENTATION_REQUEST", "PROFILE_ENV", "SESSION_KEY",
           "get", "current", "set_current", "session_root",
           "orientation_request_for", "orient_on_upload",
           "IMAGE_SUFFIXES"]
