"""Which application is running on the core: its name, its distribution and
its package, handed in by the application (BACKLOG INF-33, the app split's
step 13; from step 11 (c)'s review, ``module_work/inf32/CHECK_STEP11C.md``
section 12).

The core names no application. Where it used to write GeotechStaffEngineer's
name into another application's files and records, it now asks here:

- the run record's ``app_version`` (:mod:`aec_agent_core.activity_log`): the
  version of :func:`dist`;
- the default signer of a markup and the words that make a markup the app's
  own (:mod:`aec_agent_core.document_tools`), the default author of a Word
  file (:mod:`aec_agent_core.docx_renderer`) and the review suite's signer
  (:mod:`aec_agent_core.review_eval.runner`): :func:`name`;
- the messages of :mod:`aec_agent_core.runtime_check`: :func:`package`.

An application hands its details in once, when its package loads, through
the hook ``app.details`` (:mod:`aec_agent_core.hooks`)::

    from aec_agent_core.app_details import hand_in
    hand_in(name="My App", dist="my-app", package="my_app")

Nothing handed in, each place uses :data:`DEFAULTS` (a neutral name, no
distribution, so no version is recorded).
"""

from __future__ import annotations

from typing import Dict, Optional

__all__ = ["HOOK", "DEFAULTS", "hand_in", "details", "name", "dist",
           "package", "markup_author", "own_mark_words"]

#: The hook an application fills: a callable returning a dict with any of
#: ``name``, ``dist`` and ``package``.
HOOK = "app.details"

#: What each place uses when no application handed its details in.
DEFAULTS: Dict[str, Optional[str]] = {
    "name": "AEC agent",
    "dist": None,
    "package": "aec_agent_core",
}


def hand_in(*, name: str, dist: Optional[str] = None,
            package: Optional[str] = None) -> None:
    """Fill the ``app.details`` hook with these values (a later call
    replaces an earlier one)."""
    from aec_agent_core.hooks import set_hook
    values = {"name": str(name), "dist": dist, "package": package}

    def _details() -> dict:
        return dict(values)
    set_hook(HOOK, _details)


def details() -> Dict[str, Optional[str]]:
    """The application's details: what it handed in over :data:`DEFAULTS`.
    Never raises."""
    out = dict(DEFAULTS)
    try:
        from aec_agent_core.hooks import get_hook
        fn = get_hook(HOOK)
        got = fn() if fn is not None else None
        if isinstance(got, dict):
            out.update({k: v for k, v in got.items()
                        if k in DEFAULTS and v is not None})
    except Exception:                                  # noqa: BLE001
        pass
    return out


def name() -> str:
    """The application's name, as a reader sees it."""
    return str(details()["name"] or DEFAULTS["name"])


def dist() -> Optional[str]:
    """The application's distribution on PyPI (None: none named)."""
    return details()["dist"]


def package() -> str:
    """The import name the application's messages give."""
    return str(details()["package"] or DEFAULTS["package"])


def markup_author() -> str:
    """Who a markup is signed by when nobody is named: the application's
    name, marked as a draft."""
    return f"{name()} (AI draft)"


def own_mark_words() -> tuple:
    """Words in a markup's author that make it one this application wrote."""
    return (name(), "(AI draft)")
