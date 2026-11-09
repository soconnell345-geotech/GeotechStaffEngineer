"""Where this app's review-suite documents come from, handed to the core's
suite framework through hooks (BACKLOG INF-32, the app split's step 11 (c);
``aec_agent_core.review_eval.documents``). Before the move the documents
module did each of these itself.

- :func:`document_dirs` (``review_eval.document_dirs``): the source
  checkout's folders (the Mecklenburg ground truth, the reference library's
  PDFs, the sample PDFs, the checkout's root);
- :func:`fetch_document` (``review_eval.fetch_document``): the SharePoint
  ``primary_references/`` fetcher the app already reads UFCs from;
- :func:`fixture` (``review_eval.fixture``): this app's synthetic
  geotechnical report (``report_fixture``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    # funhouse_agent/review_eval/host.py -> the checkout (where the documents
    # module computed it from before it moved into the core).
    return Path(__file__).resolve().parents[2]


def document_dirs() -> List[Path]:
    root = _repo_root()
    return [root / "module_work" / "drawing_ground_truth" / "mecklenburg",
            root / "geotech-references" / "docs", root / "sample_pdfs", root]


def fetch_document(name: str) -> Optional[str]:
    try:
        # The app registers its SharePoint primary_references/ fetcher when
        # it builds an agent; a suite run looks documents up BEFORE that.
        from webapp.core import _register_reference_fetcher
        _register_reference_fetcher()
    except Exception:
        pass
    from funhouse_agent import reference_docs
    return reference_docs.fetch(name)


def fixture(name: str) -> Optional[bytes]:
    if name == "report":
        from funhouse_agent.review_eval.report_fixture import (
            build_synthetic_extraction_report)
        return build_synthetic_extraction_report().pdf
    return None
