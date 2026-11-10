"""``aec-document-review``: start the page with Streamlit (``streamlit run``
of :mod:`aec_document_review.entry`), passing any further arguments to
Streamlit (for example ``--server.port 8502``)."""

from __future__ import annotations

import os
import sys


def entry_path() -> str:
    """The Streamlit script's path (:mod:`aec_document_review.entry`)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "entry.py")


def main(argv=None) -> int:
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", entry_path(),
                *(sys.argv[1:] if argv is None else list(argv))]
    return stcli.main()


if __name__ == "__main__":
    sys.exit(main())
