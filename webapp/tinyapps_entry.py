"""The Tiny Apps entry point: the two-page app on CfA's Azure App Service.

Tiny Apps starts Streamlit on the wrapper repo's ``app.py``, which is a few
lines ending in::

    from webapp.tinyapps_entry import main
    main()

``main()`` marks the process as a Tiny Apps deployment, installs the Prompter
engine from the deployment's settings, and runs the two pages declared in
:mod:`webapp.pages_entry` — Document Review at the root, GeotechStaffEngineer
at ``/geotech`` — both over the one ``webapp/app.py`` shell. Upgrading the
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

from webapp import pages_entry

# The page machinery lives in pages_entry (host-neutral); these names stay
# importable from here for callers and tests that knew them first.
app_path = pages_entry.app_path
page_runner = pages_entry.page_runner
build_navigation = pages_entry.build_navigation

DEPLOYMENT_NAME = "tinyapps"

_ENGINE_REGISTERED = False


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


def main() -> None:
    """Run the two-page app in the current Streamlit script context."""
    mark_deployment()
    register_engine()
    pages_entry.run()


if __name__ == "__main__":
    main()
