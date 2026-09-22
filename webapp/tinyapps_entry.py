"""The Tiny Apps entry point: two pages, one app.

Tiny Apps (CfA's Azure App Service hosting) starts Streamlit on the wrapper
repo's ``app.py``, which is a few lines ending in::

    from webapp.tinyapps_entry import main
    main()

``main()`` marks the process as a Tiny Apps deployment, installs the Prompter
engine from the deployment's settings, and runs a two-page navigation:

* **Document Review** (``/review``, the default) — a general document-review
  agent for architects, engineers and construction managers: looks at the
  pages, hands back Word memos and marked-up PDFs.
* **GeotechStaffEngineer** (``/geotech``) — the geotechnical staff engineer
  exactly as on Databricks.

Both pages run the SAME ``webapp/app.py`` shell under an
:class:`~webapp.profiles.AppProfile`; nothing is duplicated. Upgrading the
deployed app = bumping the ``geotech-staff-engineer`` pin in ``packages.txt``
and asking the App Services team to sync.

Environment (the deployment's settings, read by :mod:`webapp.tinyapps_settings`
from env, Key Vault or a local ``.env``)::

    PROMPTER_URL / PROMPTER_MODEL / PROMPTER_API_KEY / PROMPTER_CA_BUNDLE
    GRAPH_TENANT_ID / GRAPH_CLIENT_ID / GRAPH_CLIENT_SECRET / SHAREPOINT_SITE_URL
    GEOTECH_WEBAPP_DATA          where conversations live (the wrapper sets it)
    DEV_IDENTITY                 local stand-in for the IIS identity header
"""

from __future__ import annotations

import os
import runpy

from webapp import profiles

DEPLOYMENT_NAME = "tinyapps"

_ENGINE_REGISTERED = False


def app_path() -> str:
    """Absolute path of the packaged Streamlit script (``webapp/app.py``)."""
    import webapp
    return os.path.join(os.path.dirname(os.path.abspath(webapp.__file__)),
                        "app.py")


def mark_deployment() -> None:
    """Set ``GEOTECH_DEPLOYMENT=tinyapps`` (idempotent): no personal API key
    is read or named anywhere; the only engine is the deployment's Prompter."""
    os.environ.setdefault("GEOTECH_DEPLOYMENT", DEPLOYMENT_NAME)


def register_engine() -> bool:
    """Install the Prompter builder once per process. Returns True when the
    settings were present; False leaves the app to show its banner."""
    global _ENGINE_REGISTERED
    if _ENGINE_REGISTERED:
        return True
    from webapp import tinyapps_engine
    ok = tinyapps_engine.register()
    _ENGINE_REGISTERED = bool(ok)
    return ok


def page_runner(profile: profiles.AppProfile):
    """A page function for ``st.Page``: records the profile, runs the shell.

    ``runpy`` executes ``webapp/app.py`` as a script in a fresh namespace on
    every rerun, exactly as Streamlit itself would run it — the shell reads
    the profile from session state and shapes itself accordingly.
    """
    def _run() -> None:
        profiles.set_current(profile.name)
        runpy.run_path(app_path(), run_name="__main__")
    _run.__name__ = profile.name
    _run.__doc__ = profile.caption
    return _run


def build_navigation():
    """The ``st.navigation`` object for the two pages (Document Review first,
    and the default)."""
    import streamlit as st
    pages = [
        st.Page(page_runner(profiles.DOCUMENT_REVIEW),
                title=profiles.DOCUMENT_REVIEW.title,
                icon=profiles.DOCUMENT_REVIEW.icon,
                url_path=profiles.DOCUMENT_REVIEW.url_path, default=True),
        st.Page(page_runner(profiles.GEOTECH),
                title=profiles.GEOTECH.title, icon=profiles.GEOTECH.icon,
                url_path=profiles.GEOTECH.url_path),
    ]
    return st.navigation(pages)


def main() -> None:
    """Run the two-page app in the current Streamlit script context."""
    mark_deployment()
    register_engine()
    import streamlit as st
    # The ONE set_page_config of the run — the shell skips its own when it
    # sees this flag (a second call raises).
    st.set_page_config(page_title="Document Review", page_icon="📄",
                       layout="wide")
    st.session_state["_page_config_set"] = True
    build_navigation().run()


if __name__ == "__main__":
    main()
