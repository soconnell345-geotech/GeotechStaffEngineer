"""The page, host-neutral (copied from GeotechStaffEngineer's
``webapp/pages_entry.py`` at the app split's step 13, BACKLOG INF-33).

Since the copy this package serves ONE page, Document Review, at the app
root: Tiny Apps through :mod:`aec_document_review.tinyapps_entry`, a laptop
through ``aec-document-review`` (:mod:`aec_document_review.launch`) or
``streamlit run`` of :mod:`aec_document_review.entry`. Nothing here knows
which host it is on; the engine, the identity and the storage are settled by
whoever starts the process. The page runs the shell, ``app.py``, under its
profile (:mod:`aec_document_review.profiles`).
"""

from __future__ import annotations

import os
import runpy

from aec_document_review import profiles


def app_path() -> str:
    """Absolute path of the packaged Streamlit shell (``aec_document_review/app.py``)."""
    import aec_document_review
    return os.path.join(os.path.dirname(os.path.abspath(aec_document_review.__file__)),
                        "app.py")


def page_runner(profile: profiles.AppProfile):
    """A page function for ``st.Page``: sets the page config, records the
    profile, runs the shell.

    ``runpy`` executes ``aec_document_review/app.py`` as a script in a fresh namespace on
    every rerun, exactly as Streamlit itself would run it — the shell reads
    the profile from session state and shapes itself accordingly.
    """
    def _run() -> None:
        import streamlit as st
        # The ONE set_page_config of the run, with THIS page's title and
        # icon (so the browser tab follows the page); the shell skips its own
        # when it sees the flag — a second call raises.
        st.set_page_config(page_title=profile.title, page_icon=profile.icon,
                           layout="wide")
        st.session_state["_page_config_set"] = True
        profiles.set_current(profile.name)
        # a crash of the page is kept in the open conversation's record
        # (BACKLOG EV-19); raised on unchanged
        from aec_document_review import page_errors
        with page_errors.recording():
            runpy.run_path(app_path(), run_name="__main__")
    _run.__name__ = profile.name
    _run.__doc__ = profile.caption
    return _run


def build_navigation():
    """The ``st.navigation`` object for the one page (Document Review, the
    default, served at the app root: it gets no ``url_path``). One page, so
    Streamlit draws no page list."""
    import streamlit as st
    pages = [
        st.Page(page_runner(profiles.DOCUMENT_REVIEW),
                title=profiles.DOCUMENT_REVIEW.title,
                icon=profiles.DOCUMENT_REVIEW.icon, default=True),
    ]
    return st.navigation(pages)


def run() -> None:
    """Run the page in the current Streamlit script context."""
    build_navigation().run()


if __name__ == "__main__":
    run()
