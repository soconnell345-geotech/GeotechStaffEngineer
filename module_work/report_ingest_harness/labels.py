"""The hand-labelled page types, mapped onto the ingest's label vocabulary.

The reading and the mapping live in :mod:`report_ingest.corpus`, which takes
the spreadsheet's path as an argument so the cluster can read the same file
from a Volume. This module is that reader bound to the repo's copy, under
the names the WP0 and WP1 scorecards already use.

The ground truth is a spreadsheet in the gitignored corpus folder: 4,300
pages of 15 reports, labelled by hand. Its first sheet is a summary naming
each label sheet's report BY FILE NAME; every other sheet has one row per
page with the page's text and its ``page_type``.

PRIVACY. Sheet names, report names and engineer names in that workbook are
private and must never reach a tracked file -- the sheet names included, which
carry a firm and a city. The sheet-to-ID matching is therefore derived at
RUNTIME and cached into ``raw/labels_map.json``; everything returned or
printed here is an ID, a sheet index, a page count or a label.

:data:`RAW_TO_LABEL` is where the two vocabularies meet. It is deliberately
total over the 23 raw strings the spreadsheet uses: an unknown string raises
rather than being dropped, because a silently ignored label is a silently
wrong score.
"""

from __future__ import annotations

from typing import Dict, List

from report_ingest.corpus import (  # noqa: F401  - re-exported by name
    LABELS, RAW_TO_LABEL, PageLabel, SheetMatch, UnknownPageType, to_label,
)

from module_work.report_ingest_harness.corpus import (
    CORPUS, LABELS_XLSX, raw_available,
)


def labels_available() -> bool:
    return raw_available() and LABELS_XLSX.is_file()


def sheet_map(refresh: bool = False) -> List[SheetMatch]:
    """Match every label sheet to a corpus ID, caching into ``raw/``."""
    return CORPUS.sheet_map(refresh=refresh)


def mapped_ids() -> List[str]:
    """Corpus IDs that have hand labels, in ID order."""
    return CORPUS.mapped_ids()


def labels_for(rid: str) -> List[PageLabel]:
    """The hand labels for one report, 0-based, in page order."""
    return CORPUS.labels_for(rid)


def label_counts() -> Dict[str, int]:
    """Hand-label counts per mapped label, over every matched report."""
    return CORPUS.label_counts()
