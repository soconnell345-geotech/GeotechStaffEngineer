"""The 38-report corpus at its place in this repo.

The loading itself lives in :mod:`report_ingest.corpus`, which takes
directories as arguments, because the same corpus has to be read on the
cluster where this repo does not exist. This module is that loader bound to
the repo's own gitignored paths, with the names the WP0 measurement scripts
already use. Everything it does is documented there; what is here is where.

PRIVACY. Everything read lives under :data:`RAW_DIR`, which is gitignored
(``module_work/field_feedback/**/raw/``). A report is an ID (R01-R38)
everywhere. :attr:`ReportInfo.private_name` holds the manifest's source-file
stem because the label-sheet matching needs it, and nothing prints it.

``open_report(rid, di=...)`` has three modes, because planlens'
``text_source`` REPLACES the PDF text layer on every page it covers rather
than filling gaps: ``"none"`` is the text layer alone, ``"all"`` hands every
covered page to Azure Document Intelligence, and ``"auto"`` (the default)
attaches DI only to the pages ``pages_needing_ocr`` names. Measured
2026-09-16, counting pages whose text came out ``azure_di``:

| report | pages | DI covers | read by DI, ``auto`` | ``all`` | s, ``auto`` | s, ``all`` |
|---|---|---|---|---|---|---|
| R28 | 455 | 455 | 28 | 455 | 11 | 63 |
| R15 | 202 | 202 | 78 | 202 | 9 | 13 |
| R13 | 197 | 197 | 197 | 197 | 9 | 9 |

R28 is the case the table exists for: 28 of its 455 pages need help, and
``all`` would throw away the exact embedded text on the other 427 to buy
optical text for those 28, at six times the wall clock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from report_ingest.corpus import (  # noqa: F401  - re-exported by name
    DI_MODES, Corpus, PageLabel, ReportInfo, SheetMatch, text_source_counts,
)

#: The gitignored corpus root. Nothing under here may be committed.
RAW_DIR = (Path(__file__).resolve().parents[1]
           / "field_feedback" / "2026-09-16_report-ingest" / "raw")
CORPUS_DIR = RAW_DIR / "corpus"
DI_DIR = RAW_DIR / "di"
#: Where the harness writes renders and scorecards for a human to look at.
CHECKS_DIR = RAW_DIR / "checks"
MANIFEST = CORPUS_DIR / "MANIFEST.md"
DI_MANIFEST = DI_DIR / "MANIFEST.md"
LABELS_XLSX = CORPUS_DIR / "trial_pages_working_r2.xlsx"
#: Runtime cache of the label-sheet-to-ID matching (derived, stays private).
LABELS_MAP_JSON = RAW_DIR / "labels_map.json"

#: The corpus, bound to this repo. One instance, so the manifest and the
#: sheet map are parsed once per process.
CORPUS = Corpus(CORPUS_DIR, di_dir=DI_DIR, labels_xlsx=LABELS_XLSX,
                cache_dir=RAW_DIR)


def raw_available() -> bool:
    """Is the private corpus present on this machine?"""
    return MANIFEST.is_file() and CORPUS_DIR.is_dir()


def _require_raw() -> None:
    if not raw_available():
        raise FileNotFoundError(
            f"the report-ingest corpus is not on this machine ({RAW_DIR} is "
            f"gitignored and exists only where the owner put it)")


def list_reports(refresh: bool = False) -> List[ReportInfo]:
    """Every corpus report, in ID order, parsed from the private manifest."""
    _require_raw()
    return CORPUS.list_reports(refresh=refresh)


def report(rid: str) -> ReportInfo:
    """The manifest row for one ID."""
    _require_raw()
    return CORPUS.report(rid)


def pdf_path(rid: str) -> Path:
    """The PDF for one ID (always ``<ID>.pdf`` inside ``raw/corpus``)."""
    _require_raw()
    return CORPUS.pdf_path(rid)


def di_manifest() -> Dict[str, Dict[str, Any]]:
    """ID to its DI manifest row; no JSON is opened."""
    return CORPUS.di_manifest_rows()


def di_file(rid: str) -> Path:
    return CORPUS.di_file(rid)


def has_di(rid: str) -> bool:
    """Is a usable DI result on disk for this ID?"""
    return CORPUS.has_di(rid)


def di_page_count(rid: str) -> Optional[int]:
    """Pages the DI result covers, from the manifest."""
    row = CORPUS.di_manifest_rows().get(rid)
    if row is None or not row["ok"]:
        return None
    return row["di_pages"] or None


def load_di(rid: str) -> Optional[dict]:
    """The gunzipped DI result for one ID, or None with a warning."""
    return CORPUS.load_di(rid)


def open_report(rid: str, di: str = "auto", *, warn: bool = True,
                ocr_pages: Optional[Sequence[int]] = None) -> Any:
    """Open one corpus report as a planlens ``Document``."""
    _require_raw()
    return CORPUS.open_report(rid, di, warn=warn, ocr_pages=ocr_pages)
