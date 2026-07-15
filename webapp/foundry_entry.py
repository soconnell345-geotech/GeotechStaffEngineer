"""Palantir Foundry entry stub for the geotech web chat app.

Foundry Code Workspaces publishes a Streamlit app by pointing at a Python file
in the workspace repo. Keep the WORKSPACE file to two lines and let the pip
package own everything else::

    # app.py  (the file named when publishing the application)
    from webapp.foundry_entry import main
    main()

``main()`` locates the installed :mod:`webapp` package's ``app.py`` and
executes it in this Streamlit script run — so upgrading the app on Foundry is
just bumping the ``geotech-staff-engineer`` version in the Libraries panel.

Setup, model RIDs, and the proxy smoke test live in ``docs/FOUNDRY.md``.
"""

from __future__ import annotations

import os
import runpy


def app_path() -> str:
    """Absolute path of the packaged Streamlit script (``webapp/app.py``)."""
    import webapp
    return os.path.join(os.path.dirname(os.path.abspath(webapp.__file__)),
                        "app.py")


def main() -> None:
    """Run the packaged app in the current Streamlit script context."""
    runpy.run_path(app_path(), run_name="__main__")


if __name__ == "__main__":
    main()
