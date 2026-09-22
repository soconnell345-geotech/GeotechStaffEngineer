"""App profiles — the pages of the Tiny Apps build, one webapp underneath.

The owner's rule for the Tiny Apps build (2026-09-21): two pages, NO second
copy of the app. Page one is a general document-review agent for anyone who
designs or builds — the planlens front door; page two is GeotechStaffEngineer
as it has always been. Both run the SAME ``webapp/app.py`` shell, the same
persistence, the same engine, the same file and vision tools. What differs is
an :class:`AppProfile`: the title on the page, the agent the shell builds
(prompt, tool scope, sub-agents), whether the specialist picker applies,
whether a fresh upload gets an automatic orientation turn, and where that
person's conversations for that page live on disk.

The current profile is chosen by the page function in
:mod:`webapp.tinyapps_entry` (``st.navigation``), or by the
``GEOTECH_APP_PROFILE`` environment variable for a single-page launch, and
defaults to the geotech profile — so the Databricks launcher and a plain
``streamlit run webapp/app.py`` behave exactly as before.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from webapp import core
from webapp.identity import Identity

#: Env var naming the profile for a single-page launch.
PROFILE_ENV = "GEOTECH_APP_PROFILE"
#: The session-state key the entry's page function sets before the shell runs.
SESSION_KEY = "_app_profile"

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


@dataclass(frozen=True)
class AppProfile:
    """One page of the app.

    ``build_kwargs`` are handed to :func:`webapp.core.build_agent` and on to
    ``build_deep_agent`` — the geotech profile passes none and gets the
    agent it always had; the document-review profile passes an empty module
    scope, no reference sub-agent and its own prompt.
    """
    name: str
    title: str
    icon: str
    caption: str
    #: The path segment in the browser (``/geotech``). Empty for the DEFAULT
    #: page, which Streamlit serves at the app root and nowhere else.
    url_path: str
    #: Whether the sidebar "Agent" picker (reviewer specialists) applies.
    specialists: bool = True
    #: Whether a fresh upload triggers an automatic orientation turn.
    orientation: bool = False
    orientation_request: str = ORIENTATION_REQUEST
    _build_kwargs: Dict[str, object] = field(default_factory=dict)

    def build_kwargs(self) -> Dict[str, object]:
        """The ``build_deep_agent`` overrides for this page (a fresh dict)."""
        kw = dict(self._build_kwargs)
        if kw.pop("_document_review_prompt", False):
            from funhouse_agent.deep.prompt import build_document_review_prompt
            kw["system_prompt"] = build_document_review_prompt(
                memory_enabled=bool(kw.get("enable_memory")))
        return kw


GEOTECH = AppProfile(
    name="geotech",
    title="GeotechStaffEngineer",
    icon="⛰️",
    caption=("An LLM agent that drives industry-standard geotechnical analysis "
             "methods. Research/analysis aid — not a design deliverable."),
    url_path="geotech",
    specialists=True,
    orientation=False,
)

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
    },
)

PROFILES: Dict[str, AppProfile] = {p.name: p for p in (DOCUMENT_REVIEW, GEOTECH)}
DEFAULT = GEOTECH


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
    if profile is DEFAULT:
        return root
    return os.path.join(root, "pages", profile.name)


__all__ = ["AppProfile", "GEOTECH", "DOCUMENT_REVIEW", "PROFILES", "DEFAULT",
           "ORIENTATION_REQUEST", "PROFILE_ENV", "SESSION_KEY",
           "get", "current", "set_current", "session_root"]
