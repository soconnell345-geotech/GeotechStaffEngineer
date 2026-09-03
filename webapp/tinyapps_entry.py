"""Data.State TinyApps entry stub for the geotech web chat app.

TinyApps (CfA Azure App Service hosting) launches Streamlit via a ``run.sh``
that runs a single file in the wrapper repo. Keep that repo's ``app.py`` to a
few lines — Key Vault secret loading (their required pattern) plus::

    from webapp.tinyapps_entry import main
    main()

``main()`` marks the process as a TinyApps deployment and executes the
packaged app. Upgrading the deployed app = bumping the
``geotech-staff-engineer`` pin in ``packages.txt`` and asking the App
Services team to sync.

Wrapper-repo templates + the pilot plan live in ``tinyapps/``.

Environment (set from Key Vault secrets in the wrapper app.py, per the
Tiny Apps User Guide §3.4):
    GEOTECH_PROMPTER_API_KEY   the pilot's Prompter key (engine wiring lands
                               once the key-auth client details are known —
                               office-hours question #1 in TINYAPPS.md)
    GEOTECH_SHAREPOINT_*       optional permanent-storage credentials
"""

from __future__ import annotations

import os
import runpy

DEPLOYMENT_NAME = "tinyapps"


def app_path() -> str:
    """Absolute path of the packaged Streamlit script (``webapp/app.py``)."""
    import webapp
    return os.path.join(os.path.dirname(os.path.abspath(webapp.__file__)),
                        "app.py")


def main() -> None:
    """Run the packaged app in the current Streamlit script context.

    Marks the process as a TinyApps deployment BEFORE the app runs:
    - no ANTHROPIC key path is offered or mentioned (same posture as the
      Foundry mode);
    - the Databricks launcher/driver-proxy machinery is irrelevant here —
      TinyApps is a plain Azure App Service behind Entra ID, so websockets
      behave normally.
    """
    os.environ.setdefault("GEOTECH_DEPLOYMENT", DEPLOYMENT_NAME)
    runpy.run_path(app_path(), run_name="__main__")


if __name__ == "__main__":
    main()
