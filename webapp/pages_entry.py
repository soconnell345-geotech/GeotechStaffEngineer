"""The two-page app, host-neutral: Document Review and GeotechStaffEngineer.

Every host that runs this app runs the SAME two pages over the SAME shell —
Tiny Apps through :mod:`webapp.tinyapps_entry`, Databricks through
:func:`webapp.databricks_launcher.run_on_databricks` (which boots this file
by default since 5.26), and a laptop through ``streamlit run
webapp/pages_entry.py``. Nothing here knows which host it is on; the engine,
the identity and the storage are settled by whoever starts the process, and
this file only declares the pages:

* **Document Review** (the app root, the default) — a general document-review
  agent for architects, engineers and construction managers: looks at the
  pages, hands back Word memos and marked-up PDFs.
* **GeotechStaffEngineer** (``/geotech``) — the geotechnical staff engineer.

Both pages run ``webapp/app.py`` under an :class:`~webapp.profiles.AppProfile`
(see :mod:`webapp.profiles`); ``streamlit run webapp/app.py`` on its own is
still the single geotech page it always was.
"""

from __future__ import annotations

import os
import runpy

from webapp import profiles


def app_path() -> str:
    """Absolute path of the packaged Streamlit shell (``webapp/app.py``)."""
    import webapp
    return os.path.join(os.path.dirname(os.path.abspath(webapp.__file__)),
                        "app.py")


def page_runner(profile: profiles.AppProfile):
    """A page function for ``st.Page``: sets the page config, records the
    profile, runs the shell.

    ``runpy`` executes ``webapp/app.py`` as a script in a fresh namespace on
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
        runpy.run_path(app_path(), run_name="__main__")
    _run.__name__ = profile.name
    _run.__doc__ = profile.caption
    return _run


def build_navigation():
    """The ``st.navigation`` object for the two pages (Document Review first,
    and the default — Streamlit serves the default page at the app root, so
    it gets no ``url_path``; a path on it is "Page not found")."""
    import streamlit as st
    pages = [
        st.Page(page_runner(profiles.DOCUMENT_REVIEW),
                title=profiles.DOCUMENT_REVIEW.title,
                icon=profiles.DOCUMENT_REVIEW.icon, default=True),
        st.Page(page_runner(profiles.GEOTECH),
                title=profiles.GEOTECH.title, icon=profiles.GEOTECH.icon,
                url_path=profiles.GEOTECH.url_path),
    ]
    return st.navigation(pages)


def run() -> None:
    """Run the two-page app in the current Streamlit script context."""
    build_navigation().run()


if __name__ == "__main__":
    run()
