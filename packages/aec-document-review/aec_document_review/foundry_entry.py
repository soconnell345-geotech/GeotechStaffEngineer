"""The Palantir Foundry entry point: Document Review published from a Foundry
Code Workspace (BACKLOG INF-74; modelled on GeotechStaffEngineer's
``webapp/foundry_entry.py``).

Foundry publishes a Streamlit app by pointing at a Python file in the
workspace repository. Keep that WORKSPACE file to two lines and let the
package own everything else::

    # app.py  (the file named when publishing the application)
    from aec_document_review.foundry_entry import main
    main()

``main()`` marks the process as a Foundry deployment and runs the page
declared in :mod:`aec_document_review.pages_entry`, so upgrading the app on
Foundry is bumping the ``aec-document-review`` version in the workspace's
Libraries panel and republishing.

What a Foundry deployment changes, all of it already in the page and the
shared core (the geotech page's Foundry path, copied):

- the models are Foundry model RIDs only, listed in ``GEOTECH_FOUNDRY_MODELS``
  (:mod:`aec_agent_core.engine_config`); no personal API key is read or
  named anywhere;
- with ``GEOTECH_FOUNDRY_CONVERSATION_DATASET`` set to a dataset's workspace
  alias, conversations are saved to that Foundry dataset after each turn
  and brought back from the sidebar
  (:mod:`aec_agent_core.foundry_dataset_store`, chosen by
  :mod:`aec_document_review.sharepoint_store`); without it they live in the
  app's container and are lost when it is rebuilt.

Who is viewing: a Foundry app is not told, so the page runs as on a laptop
(:mod:`aec_document_review.identity` finds no sign-in header, and no
``DEV_IDENTITY``): one shared set of conversations for everyone who can
open the app. ``docs/FOUNDRY.md`` in the project's repository says what
that means for who may open it.

Plotting code written by the model is off here unless the workspace file
sets ``GEOTECH_PLOT_CODE=1`` before ``main()``, as on Tiny Apps: one process
holds every viewer's conversations and the app's settings, a document can ask
the model for code, and the box that code runs in limits Python but is not an
operating-system boundary. The figure helper and its fixed plotting tool
(``plot_data``) stay.

A crash of the page is kept in the open conversation's record (BACKLOG
EV-19) by :func:`aec_document_review.pages_entry.page_runner`, which runs
the page inside :func:`aec_document_review.page_errors.recording`, as on
every host.
"""

from __future__ import annotations

import os

from aec_document_review import pages_entry

DEPLOYMENT_NAME = "foundry"


def mark_deployment() -> None:
    """Set ``GEOTECH_DEPLOYMENT=foundry`` (idempotent; a value already set
    is kept, as GeotechStaffEngineer's Foundry entry does), and turn
    model-written plotting code off unless ``GEOTECH_PLOT_CODE`` is already
    set (the Tiny Apps wrapper does the same on its shared host)."""
    os.environ.setdefault("GEOTECH_DEPLOYMENT", DEPLOYMENT_NAME)
    os.environ.setdefault("GEOTECH_PLOT_CODE", "0")


def main() -> None:
    """Run the page in the current Streamlit script context, marked as a
    Foundry deployment BEFORE it runs."""
    mark_deployment()
    pages_entry.run()


if __name__ == "__main__":
    main()
