"""aec-document-review: an LLM agent that reviews architecture, engineering
and construction documents, as a Streamlit page.

The app split's step 13 (BACKLOG INF-33) copied the Document Review page out
of GeotechStaffEngineer into this package. It runs on the shared core
``aec-agent-core`` and on ``planlens``, and imports nothing geotechnical.

Run it::

    aec-document-review          # streamlit run of aec_document_review/entry.py

and on Tiny Apps the wrapper's ``app.py`` calls
:func:`aec_document_review.tinyapps_entry.main`.

This package imports nothing at load beyond the standard library and the
core's hook table (itself standard library only), so ``import
aec_document_review`` never needs Streamlit or a model. Loading it hands the
core what is this application's (``aec_agent_core.hooks``), as strings
imported only when the core first asks:

- ``prompter.get_setting``, ``prompter.resolve_for``,
  ``prompter.tester_keys_possible``: the Prompter connection's settings read
  through Key Vault (:mod:`.tinyapps_settings`) and whose key serves a caller
  (:mod:`.tinyapps_keys`);
- ``app.details``: this application's name, distribution and package (the
  run record's version, the default Word and markup authors);
- the review suite's ``review_eval.*`` hooks are filled by
  :mod:`aec_document_review.review_eval`, when the suite is loaded.

The core's chart-lines registration is left empty: Document Review draws no
standard geotechnical chart lines (the split's one deliberate fingerprint
change).
"""

__version__ = "0.1.1"

#: This application's name and distribution, handed to the core.
APP_NAME = "Document Review"
APP_DIST = "aec-document-review"
APP_PACKAGE = "aec_document_review"

from aec_agent_core.hooks import set_hook as _set_hook  # noqa: E402

_set_hook("prompter.get_setting",
          "aec_document_review.tinyapps_settings:get_setting")
_set_hook("prompter.resolve_for",
          "aec_document_review.tinyapps_keys:resolve_for")
_set_hook("prompter.tester_keys_possible",
          "aec_document_review.tinyapps_keys:tester_keys_possible")

from aec_agent_core.app_details import hand_in as _hand_in  # noqa: E402

_hand_in(name=APP_NAME, dist=APP_DIST, package=APP_PACKAGE)

__all__ = ["__version__", "APP_NAME", "APP_DIST", "APP_PACKAGE"]
