"""The Streamlit script that runs Document Review on any host but Tiny Apps
(where the wrapper's ``app.py`` calls :func:`aec_document_review.
tinyapps_entry.main`)::

    streamlit run "$(python -c 'import aec_document_review.entry as e; print(e.__file__)')"

or, the same, ``aec-document-review`` (:mod:`aec_document_review.launch`).
A local ``.env`` is read first, as on Tiny Apps; with no Prompter settings
the page shows its "No model configured" banner.
"""

from aec_document_review import tinyapps_entry

tinyapps_entry.load_local_settings()
tinyapps_entry.register_engine()
from aec_document_review import pages_entry  # noqa: E402

pages_entry.run()
